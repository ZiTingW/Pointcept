"""Simplified PointTransformer V3 segmentor."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields, replace
from typing import Dict, Tuple

import torch
import torch.nn as nn

from pointcept.models.point_transformer_v3.point_transformer_v3m2_sonata import (
    PointTransformerV3,
)
from pointcept.models.utils.structure import Point


def _ensure_tuple(value):
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return value


@dataclass
class BackboneConfig:
    """Configuration for the :class:`PointTransformerV3` backbone."""

    in_channels: int = 9
    order: Tuple[str, ...] = ("z", "z-trans", "hilbert", "hilbert-trans")
    stride: Tuple[int, ...] = (2, 2, 2, 2)
    enc_depths: Tuple[int, ...] = (3, 3, 3, 12, 3)
    enc_channels: Tuple[int, ...] = (48, 96, 192, 384, 512)
    enc_num_head: Tuple[int, ...] = (3, 6, 12, 24, 32)
    enc_patch_size: Tuple[int, ...] = (1024, 1024, 1024, 1024, 1024)
    dec_depths: Tuple[int, ...] = (2, 2, 2, 2)
    dec_channels: Tuple[int, ...] = (64, 96, 192, 384)
    dec_num_head: Tuple[int, ...] = (4, 6, 12, 24)
    dec_patch_size: Tuple[int, ...] = (1024, 1024, 1024, 1024)
    mlp_ratio: int = 4
    qkv_bias: bool = True
    qk_scale: float | None = None
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    drop_path: float = 0.3
    layer_scale: float | None = None
    shuffle_orders: bool = True
    pre_norm: bool = True
    enable_rpe: bool = False
    enable_flash: bool = False
    upcast_attention: bool = False
    upcast_softmax: bool = False
    traceable: bool = False
    mask_token: bool = False
    enc_mode: bool = False
    freeze_encoder: bool = False

    @classmethod
    def from_dict(cls, config: Mapping[str, object]) -> "BackboneConfig":
        """Build a configuration from a checkpoint dictionary."""

        field_names = {field.name for field in fields(cls)}
        kwargs = {}
        for key in field_names:
            if key in config:
                value = config[key]
                value = _ensure_tuple(value)
                kwargs[key] = value
        return cls(**kwargs)

    def to_kwargs(self) -> dict:
        """Convert the dataclass to a dict compatible with ``PointTransformerV3``."""

        kwargs = {}
        for field in fields(self):
            value = getattr(self, field.name)
            kwargs[field.name] = _ensure_tuple(value)
        return kwargs

    def for_probe_mode(self, probe_mode: str) -> "BackboneConfig":
        """Return a copy with ``enc_mode``/``freeze_encoder`` adapted to a probe."""

        config = replace(self)
        if probe_mode == "linear":
            config.enc_mode = True
            config.freeze_encoder = False
        elif probe_mode == "decoder":
            config.enc_mode = False
            config.freeze_encoder = True
        else:
            config.enc_mode = False
            config.freeze_encoder = False
        return config


def load_sonata_checkpoint(
    checkpoint_path: str, map_location: str | torch.device = "cpu"
) -> tuple[dict | None, Mapping[str, torch.Tensor]]:
    """Load a Sonata checkpoint and extract its config and state dict."""

    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=map_location, weights_only=True
        )
    except TypeError:  # ``weights_only`` is only available in newer PyTorch versions.
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

    config = checkpoint.get("config") if isinstance(checkpoint, Mapping) else None
    state = None
    if isinstance(checkpoint, Mapping):
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], Mapping):
            state = checkpoint["state_dict"]
        elif "model" in checkpoint and isinstance(checkpoint["model"], Mapping):
            state = checkpoint["model"]
    if state is None:
        state = checkpoint if isinstance(checkpoint, Mapping) else {}
    if "state_dict" in state and isinstance(state["state_dict"], Mapping):
        state = state["state_dict"]
    return config, state


class SimplePointTransformerV3Segmentor(nn.Module):
    """Minimal wrapper around :class:`PointTransformerV3` for segmentation."""

    def __init__(
        self,
        num_classes: int = 13,
        probe_mode: str = "linear",
        backbone_config: BackboneConfig | None = None,
    ) -> None:
        super().__init__()
        if backbone_config is None:
            backbone_config = BackboneConfig()
        self.probe_mode = probe_mode
        self.backbone_config = backbone_config.for_probe_mode(probe_mode)
        config_dict = self.backbone_config.to_kwargs()
        self.backbone = PointTransformerV3(**config_dict)
        if self.backbone_config.enc_mode:
            self.backbone_out_channels = sum(self.backbone_config.enc_channels)
        else:
            self.backbone_out_channels = self.backbone_config.dec_channels[0]
        self.seg_head = nn.Linear(self.backbone_out_channels, num_classes)
        if probe_mode == "linear":
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        point = Point(inputs)
        if self.probe_mode == "linear":
            with torch.no_grad():
                point = self.backbone(point)
        else:
            point = self.backbone(point)
        feat = self._gather_backbone_features(point)
        return self.seg_head(feat)

    def _gather_backbone_features(self, point: Point | torch.Tensor) -> torch.Tensor:
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            return point.feat
        return point

    def train(self, mode: bool = True) -> "SimplePointTransformerV3Segmentor":
        super().train(mode)
        if self.probe_mode == "linear":
            self.backbone.eval()
        elif self.probe_mode == "decoder":
            if hasattr(self.backbone, "embedding"):
                self.backbone.embedding.eval()
            if hasattr(self.backbone, "enc"):
                self.backbone.enc.eval()
        return self

    def get_trainable_parameters(self) -> Iterable[torch.nn.Parameter]:
        for param in self.parameters():
            if param.requires_grad:
                yield param

    def load_pretrained(self, checkpoint_path: str, strict: bool = False) -> Tuple[list[str], list[str]]:
        _, state_dict = load_sonata_checkpoint(checkpoint_path)
        return self.load_pretrained_state(state_dict, strict=strict)

    def load_pretrained_state(
        self, state_dict: Mapping[str, torch.Tensor], strict: bool = False
    ) -> Tuple[list[str], list[str]]:
        """Load a pre-extracted state dict into the backbone."""

        filtered = self._filter_backbone_state(state_dict)
        missing, unexpected = self.backbone.load_state_dict(filtered, strict=strict)
        return list(missing), list(unexpected)

    def _filter_backbone_state(
        self, state_dict: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        target_keys = set(self.backbone.state_dict().keys())
        filtered: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if not isinstance(value, torch.Tensor):
                continue
            if "teacher." in key:
                continue
            name = key
            for prefix in (
                "module.student.backbone.",
                "student.backbone.",
                "module.backbone.",
                "model.backbone.",
                "backbone.",
                "module.",
                "model.",
                "student.",
            ):
                if name.startswith(prefix):
                    name = name[len(prefix) :]
            if name in target_keys:
                filtered[name] = value
        return filtered
