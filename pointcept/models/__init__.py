"""Model registry exports with optional dependency handling."""

from __future__ import annotations

import warnings
from importlib import import_module

from .builder import build_model
from .default import DefaultClassifier, DefaultSegmentor
from .modules import PointModel, PointModule

__all__ = [
    "build_model",
    "DefaultClassifier",
    "DefaultSegmentor",
    "PointModel",
    "PointModule",
]

_EXPORTED = set(__all__)
_OPTIONAL_DEPENDENCIES = {
    "wandb",
    "pointops",
    "pointops._C",
    "pointops2",
    "pointops2_cuda",
    "pointops2.pointops",
    "ocnn",
}
_SUBMODULES = [
    ".sparse_unet",
    ".point_transformer",
    ".point_transformer_v2",
    ".point_transformer_v3",
    ".stratified_transformer",
    ".spvcnn",
    ".octformer",
    ".oacnns",
    ".context_aware_classifier",
    ".point_group",
    ".sgiformer",
    ".masked_scene_contrast",
    ".point_prompt_training",
    ".sonata",
]


def _register_exports(names: list[str]) -> None:
    for name in names:
        if name not in _EXPORTED:
            __all__.append(name)
            _EXPORTED.add(name)


def _safe_import(module_name: str) -> None:
    try:
        module = import_module(module_name, package=__name__)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing in _OPTIONAL_DEPENDENCIES:
            warnings.warn(
                f"Skipping '{module_name}' because optional dependency '{missing}' is unavailable.",
                ImportWarning,
            )
            return
        raise

    exports = getattr(module, "__all__", None)
    if exports is None:
        exports = [name for name in dir(module) if not name.startswith("_")]
    for attr in exports:
        globals()[attr] = getattr(module, attr)
    _register_exports(list(exports))


for _module_name in _SUBMODULES:
    _safe_import(_module_name)


del _module_name
del _register_exports
del _safe_import
del _EXPORTED
del _OPTIONAL_DEPENDENCIES
del _SUBMODULES
