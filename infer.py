"""Single-image inference and ONNX export."""

import argparse

import cv2
import numpy as np
import torch

from src.model import ChromaGatedFusion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default="enhanced.png")
    args = parser.parse_args()

    state = torch.load(args.ckpt, map_location="cpu")
    model = ChromaGatedFusion()
    model.load_state_dict(state["model"])
    model.eval()

    img = cv2.imread(args.image)[:, :, ::-1] / 255.0
    tensor = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().unsqueeze(0)
    with torch.no_grad():
        pred, _ = model(tensor)
    out = (pred[0].permute(1, 2, 0).numpy()[:, :, ::-1] * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(args.out, out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
