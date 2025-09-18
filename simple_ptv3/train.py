"""Command line training interface for the simplified PTv3 segmentor."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import DatasetConfig, S3DISSimpleDataset, simple_collate_fn
from metrics import SegmentationMetric
from model import (
    BackboneConfig,
    SimplePointTransformerV3Segmentor,
    load_sonata_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=str, required=True, help="Path to processed S3DIS")
    parser.add_argument(
        "--train-areas",
        nargs="+",
        default=["Area_1", "Area_2", "Area_3", "Area_4", "Area_6"],
        help="Training splits (directories) to use.",
    )
    parser.add_argument(
        "--val-area", type=str, default="Area_5", help="Validation split directory"
    )
    parser.add_argument("--num-points", type=int, default=65536, help="Points per room")
    parser.add_argument("--grid-size", type=float, default=0.02, help="Voxel size for grid coords")
    parser.add_argument(
        "--probe-mode",
        choices=["linear", "decoder", "finetune"],
        default="linear",
        help="Training regime.",
    )
    parser.add_argument("--pretrained", type=str, default="", help="Path to Sonata weights")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-3, help="AdamW learning rate")
    parser.add_argument("--weight-decay", type=float, default=2e-2, help="Weight decay")
    parser.add_argument("--num-workers", type=int, default=4, help="Data loader workers")
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision")
    parser.add_argument(
        "--enable-flash",
        action="store_true",
        help="Enable FlashAttention kernels if the environment supports them.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device")
    parser.add_argument("--output-dir", type=str, default="runs/simple_ptv3", help="Log directory")
    parser.add_argument("--save-best", action="store_true", help="Persist best checkpoint")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and evaluate")
    parser.add_argument(
        "--log-interval", type=int, default=10, help="Steps between progress messages"
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    train_dataset = S3DISSimpleDataset(
        DatasetConfig(
            data_root=args.data_root,
            areas=args.train_areas,
            num_points=args.num_points,
            grid_size=args.grid_size,
            augment=True,
            training=True,
        )
    )
    val_dataset = S3DISSimpleDataset(
        DatasetConfig(
            data_root=args.data_root,
            areas=[args.val_area],
            num_points=args.num_points,
            grid_size=args.grid_size,
            augment=False,
            training=False,
        )
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=simple_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=simple_collate_fn,
    )
    return train_loader, val_loader


def build_model(args: argparse.Namespace) -> SimplePointTransformerV3Segmentor:
    checkpoint_cfg = None
    checkpoint_state = None
    if args.pretrained:
        cfg_dict, checkpoint_state = load_sonata_checkpoint(args.pretrained)
        if cfg_dict is not None:
            checkpoint_cfg = BackboneConfig.from_dict(cfg_dict)
    backbone_cfg = checkpoint_cfg or BackboneConfig()
    backbone_cfg.enable_flash = bool(args.enable_flash)
    model = SimplePointTransformerV3Segmentor(
        num_classes=13,
        probe_mode=args.probe_mode,
        backbone_config=backbone_cfg,
    )
    if checkpoint_state:
        missing, unexpected = model.load_pretrained_state(checkpoint_state)
        print(f"Loaded pretrained backbone from {args.pretrained}")
        if missing:
            print(f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")
    elif args.pretrained:
        missing, unexpected = model.load_pretrained(args.pretrained)
        print(f"Loaded pretrained backbone from {args.pretrained}")
        if missing:
            print(f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")
    return model


def prepare_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    prepared = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            prepared[key] = value.to(device)
    return prepared


def train_one_epoch(
    model: SimplePointTransformerV3Segmentor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch: int,
    log_interval: int,
) -> float:
    model.train(True)
    running_loss = 0.0
    for step, batch in enumerate(loader, start=1):
        prepared = prepare_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            logits = model(prepared)
            loss = F.cross_entropy(logits, prepared["segment"], ignore_index=-1)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        if step % log_interval == 0:
            print(f"Epoch {epoch:03d} | Step {step:04d}/{len(loader):04d} | Loss {loss.item():.4f}")
    return running_loss / max(len(loader), 1)


def evaluate(
    model: SimplePointTransformerV3Segmentor,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    metric = SegmentationMetric(num_classes=13, ignore_index=-1)
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            prepared = prepare_batch(batch, device)
            logits = model(prepared)
            loss = F.cross_entropy(logits, prepared["segment"], ignore_index=-1)
            preds = logits.argmax(dim=1)
            metric.add_batch(preds.cpu(), prepared["segment"].cpu())
            total_loss += loss.item()
    metrics = metric.compute()
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics


def save_checkpoint(model: SimplePointTransformerV3Segmentor, output_dir: Path, epoch: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"checkpoint-epoch{epoch:03d}.pth"
    torch.save({"model": model.state_dict(), "epoch": epoch}, path)
    print(f"Checkpoint saved to {path}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_loader, val_loader = build_dataloaders(args)
    model = build_model(args).to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    optimizer = torch.optim.AdamW(
        model.get_trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    output_dir = Path(args.output_dir)
    best_miou = -1.0

    if args.eval_only:
        metrics = evaluate(model, val_loader, device)
        print(json.dumps(metrics, indent=2))
        return

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, args.log_interval)
        metrics = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch:03d} | Train Loss {train_loss:.4f} | Val Loss {metrics['loss']:.4f} | mIoU {metrics['mIoU']:.3f}"
        )
        if args.save_best and metrics["mIoU"] > best_miou:
            best_miou = metrics["mIoU"]
            save_checkpoint(model, output_dir, epoch)

    if not args.save_best:
        save_checkpoint(model, output_dir, args.epochs)


if __name__ == "__main__":
    main()
