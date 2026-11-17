"""FlyPE: behaviour detection in Drosophila video with YOLO11."""

from __future__ import annotations

from flype.train import DEFAULT_MODEL, TRAIN_ARGS, build_train_args, train

__version__ = "0.1.0"

__all__ = ["DEFAULT_MODEL", "TRAIN_ARGS", "build_train_args", "train"]
