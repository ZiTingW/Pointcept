# Simple PointTransformer V3 for S3DIS

This folder contains a lightweight training script that focuses on semantic
segmentation on S3DIS with a Sonata-pretrained PointTransformer V3 backbone.
Only two probing setups are provided:

- **Linear probing** – freeze the entire backbone and train a linear classifier.
- **Decoder probing** – freeze the encoder (stem + downsampling blocks) and train
the decoder together with the segmentation head.

A regular fine-tuning mode is also available for completeness.

## Requirements

The implementation reuses Pointcept modules, therefore install the same
dependencies as the original project. In addition you need the preprocessed
S3DIS rooms (see the main README for preprocessing instructions) and a Sonata
pretrained checkpoint. The official weights published with the
[`facebookresearch/sonata`](https://github.com/facebookresearch/sonata) repo can
be downloaded from Hugging Face; the script reads the configuration embedded in
those checkpoints so that the backbone is instantiated with the exact training
hyper-parameters.

## Quick start

```bash
# Linear probing
python -m simple_ptv3.train \
    --data-root data/s3dis \
    --pretrained /path/to/sonata_backbone.pth \
    --probe-mode linear \
    --batch-size 4 \
    --epochs 100

# Decoder probing
python -m simple_ptv3.train \
    --data-root data/s3dis \
    --pretrained /path/to/sonata_backbone.pth \
    --probe-mode decoder \
    --batch-size 4 \
    --epochs 100
```

By default the script trains on Areas 1/2/3/4/6 and validates on Area 5. The
number of points sampled per room, the grid size used for serialization and the
optimizer settings can be customised through command line flags (check
`python -m simple_ptv3.train --help`).

To evaluate a pretrained checkpoint without further training run:

```bash
python -m simple_ptv3.train --data-root data/s3dis --pretrained checkpoint.pth --eval-only

```

FlashAttention kernels are disabled by default so that the code can run without
the optional dependency. If your environment provides FlashAttention you can
turn them back on with `--enable-flash`.

## Outputs

When `--save-best` is enabled the best performing model w.r.t validation mIoU is
stored under the directory specified by `--output-dir`. Otherwise the final
checkpoint after the last epoch is written.

Validation metrics (loss, mIoU, overall accuracy, per-class IoU) are printed at
the end of every epoch.
