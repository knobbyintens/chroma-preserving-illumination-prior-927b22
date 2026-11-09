"""Dataset helpers for paired low-light enhancement training."""

import os
from glob import glob

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class PairedLowLightDataset(Dataset):
    def __init__(self, root: str, image_size: int = 256):
        self.pairs = []
        low_dir = os.path.join(root, "low")
        high_dir = os.path.join(root, "high")
        for low_path in sorted(glob(os.path.join(low_dir, "*"))):
            name = os.path.basename(low_path)
            high_path = os.path.join(high_dir, name)
            if os.path.exists(high_path):
                self.pairs.append((low_path, high_path))
        self.image_size = image_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        low_path, high_path = self.pairs[idx]
        low = cv2.imread(low_path)[:, :, ::-1] / 255.0
        high = cv2.imread(high_path)[:, :, ::-1] / 255.0
        low = cv2.resize(low, (self.image_size, self.image_size))
        high = cv2.resize(high, (self.image_size, self.image_size))
        to_tensor = lambda x: torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).float()
        return {"low": to_tensor(low), "high": to_tensor(high)}
