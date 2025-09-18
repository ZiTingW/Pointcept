"""Lightweight S3DIS dataset loader for PointTransformer V3."""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class DatasetConfig:
    """Configuration options for :class:`S3DISSimpleDataset`."""

    data_root: str
    areas: Sequence[str]
    num_points: int | None = 65536
    grid_size: float = 0.02
    augment: bool = False
    training: bool = True


class S3DISSimpleDataset(Dataset):
    """Minimal S3DIS loader that returns tensors ready for PointTransformer V3.

    The class expects the preprocessed S3DIS directory that ships with Pointcept,
    i.e. each area contains folders for each room and every room holds
    ``coord.npy``, ``color.npy``, ``normal.npy`` and ``segment.npy`` files.
    """

    def __init__(self, config: DatasetConfig | None = None, **kwargs):
        if config is None:
            config = DatasetConfig(**kwargs)  # type: ignore[arg-type]
        if isinstance(config.areas, str):  # pragma: no cover - convenience branch
            areas: Sequence[str] = (config.areas,)
        else:
            areas = tuple(config.areas)
        self.config = config
        self.data_root = os.path.expanduser(config.data_root)
        self.areas = areas
        self.num_points = config.num_points
        self.grid_size = config.grid_size
        self.augment = config.augment and config.training
        self.training = config.training

        self.files: List[str] = []
        for area in self.areas:
            pattern = os.path.join(self.data_root, area, "*")
            rooms = sorted(glob.glob(pattern))
            self.files.extend(rooms)
        if not self.files:
            raise FileNotFoundError(
                f"No S3DIS rooms found under '{self.data_root}' for areas {self.areas}."
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict:
        path = self.files[index]
        coord = self._load_numpy(path, "coord")
        color = self._load_numpy(path, "color", fallback=np.zeros_like(coord))
        normal = self._load_numpy(path, "normal", fallback=np.zeros_like(coord))
        label = self._load_numpy(path, "segment").astype(np.int64)

        coord = coord.astype(np.float32)
        coord -= coord.mean(axis=0, keepdims=True)
        color = (color.astype(np.float32)) / 255.0
        normal = normal.astype(np.float32)

        coord, color, normal, label = self._maybe_sample(coord, color, normal, label)

        if self.augment:
            coord, normal, color = self._augment(coord, normal, color)

        feat = np.concatenate([coord, color, normal], axis=1)

        # Grid coordinates are required by PointTransformer for serialization.
        min_coord = coord.min(axis=0, keepdims=True)
        grid_coord = np.floor((coord - min_coord) / self.grid_size).astype(np.int32)

        return {
            "coord": torch.from_numpy(coord),
            "grid_coord": torch.from_numpy(grid_coord),
            "feat": torch.from_numpy(feat),
            "segment": torch.from_numpy(label.astype(np.int64)),
            "name": os.path.basename(path),
        }

    def _load_numpy(
        self,
        directory: str,
        name: str,
        fallback: np.ndarray | None = None,
    ) -> np.ndarray:
        file_path = os.path.join(directory, f"{name}.npy")
        if os.path.isfile(file_path):
            return np.load(file_path)
        if fallback is None:
            raise FileNotFoundError(f"Expected file '{file_path}' not found.")
        return fallback.copy()

    def _maybe_sample(
        self,
        coord: np.ndarray,
        color: np.ndarray,
        normal: np.ndarray,
        label: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.num_points is None:
            return coord, color, normal, label
        total = coord.shape[0]
        if total == self.num_points:
            return coord, color, normal, label
        if self.training:
            choice = np.random.choice(total, self.num_points, replace=total < self.num_points)
        else:
            if total >= self.num_points:
                choice = np.arange(self.num_points)
            else:
                repeat = self.num_points - total
                extra = np.tile(np.arange(total), int(np.ceil(repeat / total)))[:repeat]
                choice = np.concatenate([np.arange(total), extra])
        return coord[choice], color[choice], normal[choice], label[choice]

    def _augment(
        self,
        coord: np.ndarray,
        normal: np.ndarray,
        color: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        theta = np.random.uniform(0.0, 2.0 * np.pi)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rot_z = np.array(
            [[cos_theta, -sin_theta, 0.0], [sin_theta, cos_theta, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        coord = coord @ rot_z.T
        normal = normal @ rot_z.T

        scale = np.random.uniform(0.95, 1.05)
        coord *= scale

        jitter = np.clip(np.random.normal(0.0, 0.01, size=coord.shape), -0.05, 0.05).astype(
            np.float32
        )
        coord += jitter

        color = np.clip(color + np.random.normal(0.0, 0.02, color.shape), 0.0, 1.0)
        return coord, normal, color


def simple_collate_fn(batch: Iterable[dict]) -> dict:
    """Merge a list of dataset items into a batch for PointTransformer."""

    coords: List[torch.Tensor] = []
    grid_coords: List[torch.Tensor] = []
    feats: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    offsets: List[int] = []
    names: List[str] = []

    total_points = 0
    for item in batch:
        num_points = item["coord"].shape[0]
        total_points += num_points
        offsets.append(total_points)
        coords.append(item["coord"].float())
        grid_coords.append(item["grid_coord"].int())
        feats.append(item["feat"].float())
        labels.append(item["segment"].long())
        names.append(item.get("name", ""))

    merged = {
        "coord": torch.cat(coords, dim=0),
        "grid_coord": torch.cat(grid_coords, dim=0),
        "feat": torch.cat(feats, dim=0),
        "segment": torch.cat(labels, dim=0),
        "offset": torch.tensor(offsets, dtype=torch.int32),
        "name": names,
    }
    return merged
