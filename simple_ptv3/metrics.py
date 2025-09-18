"""Utility metrics for semantic segmentation."""

from __future__ import annotations

import torch


class SegmentationMetric:
    """Track segmentation accuracy and mean IoU across iterations."""

    def __init__(self, num_classes: int, ignore_index: int = -1) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        self.confusion = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64)

    @torch.no_grad()
    def add_batch(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        if prediction.shape != target.shape:
            raise ValueError("Prediction and target must share the same shape.")
        mask = target != self.ignore_index
        if not torch.any(mask):
            return
        pred = prediction[mask].view(-1)
        gt = target[mask].view(-1)
        index = gt * self.num_classes + pred
        conf = torch.bincount(index, minlength=self.num_classes ** 2)
        self.confusion += conf.view(self.num_classes, self.num_classes).cpu()

    def compute(self) -> dict:
        conf = self.confusion.float()
        tp = conf.diag()
        pos = conf.sum(dim=1)
        res = conf.sum(dim=0)
        denom = pos + res - tp
        iou = torch.where(denom > 0, tp / denom.clamp(min=1e-6), torch.zeros_like(tp))
        overall_acc = tp.sum() / conf.sum().clamp(min=1.0)
        return {
            "mIoU": iou.mean().item(),
            "per_class_iou": iou.tolist(),
            "overall_acc": overall_acc.item(),
        }
