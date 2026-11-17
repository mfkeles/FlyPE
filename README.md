# FlyPE

Behaviour detection in *Drosophila* video with a YOLO11 object detector.

FlyPE fine-tunes a YOLO11-l detector to localise a fly and classify its
behaviour in each frame. The detector is trained on motion-heatmap frames, in
which recent movement is colour-coded on top of the grey-scale image, so that
behaviours that differ only in small movements of the proboscis or legs remain
separable.

This repository currently contains the training code. Analysis scripts will be
added as individual, self-contained scripts.

Project page: <https://mfkeles.github.io/FlyPE>

## Install

FlyPE uses [uv](https://docs.astral.sh/uv/). Pick the PyTorch runtime that
matches your machine:

```bash
git clone https://github.com/mfkeles/FlyPE.git
cd FlyPE
uv sync --extra cuda    # Windows/Linux with an NVIDIA GPU
uv sync --extra metal   # macOS on Apple Silicon
uv sync --extra cpu     # CPU only
```

## Train

```bash
uv run flype-train --data /path/to/data.yaml --project runs --device 0
```

The run is written to `runs/flype/`; the best checkpoint is
`runs/flype/weights/best.pt`. `yolo11l.pt` is downloaded on first use; pass
`--model /path/to/weights.pt` to start from a local file.

| Flag | Default | Meaning |
|------|---------|---------|
| `--data` | required | YOLO `data.yaml` |
| `--model` | `yolo11l.pt` | Pretrained weights (Ultralytics name or local path) |
| `--project` | `runs` | Output directory |
| `--name` | `flype` | Run name inside the project directory |
| `--device` | `0` | `0`, `0,1`, `mps` or `cpu` |
| `--epochs`, `--batch`, `--workers` | recipe values | Override for smaller hardware |
| `--dry-run` | off | Print the resolved arguments and exit |

The same entry point is available from Python:

```python
from flype import train

best = train("/path/to/data.yaml", project="runs", device="0")
```

### Recipe

The recipe is fixed in [`src/flype/train.py`](src/flype/train.py) (`TRAIN_ARGS`).

| Setting | Value |
|---------|-------|
| Model | YOLO11-l, COCO-pretrained |
| Image size | 800 px, rectangular batches |
| Schedule | 200 epochs, early stopping with 100-epoch patience |
| Batch size | 16 |
| Optimiser | Ultralytics `auto` |
| Seed | 0 |
| Colour augmentation | brightness only (`hsv_v=0.4`); hue and saturation off |
| Geometric augmentation | scale ±50%, translate ±10%, horizontal flip 0.5 |
| Other augmentation | random erasing 0.4, RandAugment |

Hue and saturation jitter are disabled because the motion channel of a heatmap
frame is colour-coded. Ultralytics disables mosaic and mixup when rectangular
batches are used, so neither is applied.

## Dataset

Training expects a standard
[Ultralytics detection dataset](https://docs.ultralytics.com/datasets/detect/):
one image and one label file per frame, with a `data.yaml` that lists the
splits and the class names.

```
dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

```yaml
path: /path/to/dataset
train: train/images
val: val/images
test: test/images
names:
  0: ProbPumping
  1: Moving
  2: Grooming
  3: Feeding
  4: Quiescent
  5: HaltereSwitch
```

Class indices follow the order above.
