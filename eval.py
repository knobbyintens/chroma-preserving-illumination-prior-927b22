"""Evaluation script: PSNR, SSIM and chroma drift."""

import argparse

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity

from src.data import PairedLowLightDataset
from src.model import ChromaGatedFusion


def chroma_drift(a, b):
    a_lab = cv2.cvtColor((a * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    b_lab = cv2.cvtColor((b * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    return float(np.abs(a_lab[..., 1:] - b_lab[..., 1:]).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()
    state = torch.load(args.ckpt, map_location="cpu")
    model = ChromaGatedFusion()
    model.load_state_dict(state["model"])
    model.eval()

    ds = PairedLowLightDataset(args.data_root)
    psnrs, ssims, drifts = [], [], []
    for item in ds:
        with torch.no_grad():
            pred, _ = model(item["low"].unsqueeze(0))
        pred_np = pred[0].permute(1, 2, 0).numpy()
        high_np = item["high"].permute(1, 2, 0).numpy()
        mse = float(((pred_np - high_np) ** 2).mean())
        psnrs.append(10 * np.log10(1.0 / max(mse, 1e-8)))
        ssims.append(structural_similarity(high_np, pred_np, channel_axis=2, data_range=1.0))
        drifts.append(chroma_drift(pred_np, high_np))
    print(f"PSNR={np.mean(psnrs):.2f} SSIM={np.mean(ssims):.4f} chroma_drift={np.mean(drifts):.2f}")


if __name__ == "__main__":
    main()
