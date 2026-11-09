# Chroma-Preserving Low-Light Enhancement via Retinex-Guided Illumination Priors

Reference implementation and full training pipeline for chroma-preserving low-light
image enhancement with Retinex-guided illumination priors (Vela Imaging Labs, 2026).

This repository is the official code release accompanying the technical report of the
same title. It contains the complete method: illumination-prior estimation, the gated
fusion network, chroma-preservation losses, training and evaluation entry points, plus
a small unit-test suite.

## Layout

- `src/illumination_prior.py` — illumination field estimation and Retinex decomposition
- `src/model.py` — gated fusion network and chroma residual branch
- `src/losses.py` — illumination consistency and chroma-preservation losses
- `src/data.py` — dataset helpers for paired and unpaired low-light data
- `train.py` — training entry point (paired / unpaired schedules)
- `eval.py` — PSNR / SSIM / chroma-drift evaluation
- `infer.py` — single-image inference and ONNX export
- `tests/test_illumination_prior.py` — unit tests for the prior estimator

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests
python train.py --config configs/default.yaml --data-root /path/to/dataset
python eval.py --ckpt runs/default/best.pt --data-root /path/to/dataset
```

## Licence

Released under the MIT licence; see `LICENSE`.
