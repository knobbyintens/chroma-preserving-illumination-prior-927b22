"""Training entry point for chroma-preserving low-light enhancement."""

import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader

from src.data import PairedLowLightDataset
from src.losses import total_loss
from src.model import ChromaGatedFusion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", default="runs/default")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = PairedLowLightDataset(args.data_root)
    loader = DataLoader(dataset, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=2)
    model = ChromaGatedFusion(cfg["model"]["base_width"], cfg["model"]["gate_hidden"]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    weights = {"pixel": 1.0, "illum": 0.2, "chroma": cfg["train"]["chroma_weight"]}

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        running = 0.0
        for batch in loader:
            low, high = batch["low"].to(device), batch["high"].to(device)
            pred, illum = model(low)
            loss = total_loss(pred, high, illum, weights)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running += loss.item()
        print(f"epoch {epoch + 1}: loss={running / max(len(loader), 1):.4f}")
        torch.save({"model": model.state_dict(), "epoch": epoch}, os.path.join(args.out, "best.pt"))


if __name__ == "__main__":
    main()
