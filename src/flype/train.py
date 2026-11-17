"""Train the FlyPE behaviour detector.

Fine-tunes a COCO-pretrained YOLO11-l detector on a YOLO-format dataset of
motion-heatmap frames, using the fixed recipe in ``TRAIN_ARGS``.

Example
-------
    flype-train --data /path/to/data.yaml --project runs --device 0

Print the resolved arguments without training:
    flype-train --data /path/to/data.yaml --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "yolo11l.pt"

# The training recipe. Hue and saturation jitter are off because the motion
# channel of a heatmap frame is colour-coded; only brightness is jittered.
# `mosaic` and `mixup` are recorded as passed to Ultralytics, which disables
# both whenever `rect=True`.
TRAIN_ARGS: dict[str, Any] = {
    "imgsz": 800,
    "rect": True,
    "epochs": 200,
    "patience": 100,
    "batch": 16,
    "seed": 0,
    "workers": 8,
    "optimizer": "auto",
    "close_mosaic": 10,
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.4,
    "mosaic": 1.0,
    "mixup": 0.0,
    "erasing": 0.4,
    "scale": 0.5,
    "translate": 0.1,
    "fliplr": 0.5,
    "auto_augment": "randaugment",
}


def build_train_args(
    data: str | Path,
    project: str | Path = "runs",
    name: str = "flype",
    device: str = "0",
    **overrides: Any,
) -> dict[str, Any]:
    """Return the full keyword arguments passed to ``YOLO.train``.

    ``overrides`` replace entries of ``TRAIN_ARGS``; ``None`` values are ignored.
    """
    args = dict(TRAIN_ARGS)
    args.update({key: value for key, value in overrides.items() if value is not None})
    args.update(
        data=str(Path(data).expanduser().resolve()),
        project=str(Path(project).expanduser().resolve()),
        name=name,
        device=device,
    )
    return args


def train(
    data: str | Path,
    model: str = DEFAULT_MODEL,
    project: str | Path = "runs",
    name: str = "flype",
    device: str = "0",
    **overrides: Any,
) -> Path:
    """Train the detector and return the path to the best checkpoint.

    Parameters
    ----------
    data
        YOLO ``data.yaml`` describing the train/val/test splits and class names.
    model
        Pretrained weights to start from: an Ultralytics name (downloaded on
        first use) or a local ``.pt`` path.
    project, name
        The run is written to ``<project>/<name>``.
    device
        Ultralytics device string, e.g. ``"0"``, ``"0,1"``, ``"mps"`` or ``"cpu"``.
    overrides
        Replacements for entries of ``TRAIN_ARGS`` (e.g. ``batch=8``).
    """
    from ultralytics import YOLO

    args = build_train_args(data, project=project, name=name, device=device, **overrides)
    detector = YOLO(model)
    detector.train(**args)
    return Path(detector.trainer.best)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="flype-train",
        description="Train the FlyPE behaviour detector (YOLO11-l).",
    )
    parser.add_argument("--data", type=Path, required=True, help="Path to the YOLO data.yaml.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Pretrained weights: an Ultralytics name or a local .pt path (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument("--project", type=Path, default=Path("runs"), help="Output directory (default: runs).")
    parser.add_argument("--name", default="flype", help="Run name inside the project directory (default: flype).")
    parser.add_argument(
        "--device", default="0", help="Ultralytics device, e.g. 0, 0,1, mps or cpu (default: 0)."
    )
    parser.add_argument("--epochs", type=int, default=None, help=f"Override epochs (default: {TRAIN_ARGS['epochs']}).")
    parser.add_argument("--batch", type=int, default=None, help=f"Override batch size (default: {TRAIN_ARGS['batch']}).")
    parser.add_argument(
        "--workers", type=int, default=None, help=f"Override data-loader workers (default: {TRAIN_ARGS['workers']})."
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved arguments and exit.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.data.is_file():
        raise SystemExit(f"data.yaml not found: {args.data}")

    overrides = {"epochs": args.epochs, "batch": args.batch, "workers": args.workers}
    if args.dry_run:
        resolved = build_train_args(args.data, args.project, args.name, args.device, **overrides)
        print(f"model: {args.model}")
        for key in sorted(resolved):
            print(f"{key}: {resolved[key]}")
        return 0

    best = train(args.data, args.model, args.project, args.name, args.device, **overrides)
    print(f"Best checkpoint: {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
