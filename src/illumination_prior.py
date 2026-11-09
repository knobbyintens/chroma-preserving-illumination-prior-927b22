"""Illumination-prior estimation and Retinex decomposition."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def retinex_decompose(image: torch.Tensor, illumination: torch.Tensor, eps: float = 1e-3):
    """Split an image into reflectance and illumination components."""
    illum = illumination.clamp(min=eps, max=1.0)
    reflectance = image / illum
    return reflectance.clamp(0.0, 4.0), illum


class IlluminationEstimator(nn.Module):
    """Lightweight U-Net style estimator of a smooth illumination field."""

    def __init__(self, base_width: int = 32):
        super().__init__()
        w = base_width
        self.enc1 = nn.Sequential(nn.Conv2d(3, w, 3, 1, 1), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(nn.Conv2d(w, w * 2, 3, 2, 1), nn.ReLU(inplace=True))
        self.mid = nn.Sequential(
            nn.Conv2d(w * 2, w * 2, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(w * 2, w * 2, 3, 1, 1), nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Conv2d(w * 2, w, 3, 1, 1)
        self.dec1 = nn.Conv2d(w, 1, 3, 1, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(image)
        e2 = self.enc2(e1)
        m = self.mid(e2)
        d2 = F.interpolate(self.dec2(m), size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(F.relu(d2 + e1))
        return torch.sigmoid(d1).clamp(0.05, 1.0)


class GuidedPrior(nn.Module):
    """Combines the illumination field with a Retinex reflectance stat."""

    def __init__(self, gate_hidden: int = 16):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(3, gate_hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(gate_hidden, 1, 1), nn.Sigmoid(),
        )

    def forward(self, image: torch.Tensor, illumination: torch.Tensor, reflectance: torch.Tensor):
        prior = torch.cat([illumination, reflectance.mean(dim=1, keepdim=True), reflectance.max(dim=1, keepdim=True)[0]], dim=1)
        gate = self.gate(prior)
        return gate * illumination + (1.0 - gate) * prior.mean(dim=1, keepdim=True)
