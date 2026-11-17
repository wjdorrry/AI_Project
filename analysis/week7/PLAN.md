# Week 7 — Style recognition quality

## Goal

Improve style recognition so the model no longer overpredicts Streetwear and can reliably distinguish the 9 project styles:

- Elegant
- Casual
- Romantic
- Minimalist
- Sporty
- Streetwear
- Boho
- Office
- Evening

The color model from Week 6 is kept unchanged.

## Dataset plan

Target a balanced style dataset.

Large external full-body source:
- Casual
- Sporty
- Streetwear
- Office
- Evening

Collect about 300 unique training images per class in small parquet-shard batches.

Exact project-style source:
- Elegant
- Romantic
- Minimalist
- Boho

Use original labeled images as the basis, with augmentation only in the training split. Validation and test must contain original images only.

## Model A — current MultiHead ViT

- Start from the Week 6 color-improved checkpoint.
- Freeze color head.
- Train style head.
- Unfreeze only the last ViT block if needed.
- Use balanced sampling.
- Save best checkpoint by macro-F1.
- Measure per-class precision, recall and F1.
- Track Streetwear prediction rate.
- Verify color quality does not regress.

## Model B — FashionCLIP classifier

Use patrickjohncyh/fashion-clip as a fashion-aware frozen image encoder.

- Extract image embeddings.
- Train a small classifier on top of frozen embeddings.
- Compare against Model A on the exact same test split.
- Keep training lightweight enough for GTX 1650 4 GB.

## Evaluation

Primary:
- Macro-F1
- Accuracy
- Per-class F1
- Confusion matrix

Bias diagnostics:
- Distribution of predicted classes
- Streetwear false-positive count
- Top-1 confidence
- Top-1 vs Top-2 margin

Real-photo benchmark:
- 20–30 manually labeled user-style photos
- Compare old Week 6 model vs Week 7 candidates

## Selection rule

The final style model is the one with the best macro-F1 and the most balanced per-class behavior, not merely the highest overall accuracy.

## Safety for the existing system

- Do not overwrite `ml/runs/latest` automatically.
- Store each Week 7 candidate in a separate run directory.
- Replace the live model only after comparison.
- Keep a backup of the Week 6 checkpoint.
