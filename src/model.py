"""Gated fusion network for chroma-preserving low-light enhancement."""

import torch
import torch.nn as nn

from .illumination_prior import IlluminationEstimator, GuidedPrior, retinex_decompose


class ChromaGatedFusion(nn.Module):
    def __init__(self, base_width: int = 32, gate_hidden: int = 16):
        super().__init__()
        self.illum_estimator = IlluminationEstimator(base_width)
        self.prior = GuidedPrior(gate_hidden)
        self.fuse = nn.Sequential(
            nn.Conv2d(6, base_width, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(base_width, 3, 1),
        )
        self.chroma_branch = nn.Conv2d(3, 3, 1)

    def forward(self, image: torch.Tensor):
        illumination = self.illum_estimator(image)
        reflectance, illum = retinex_decompose(image, illumination)
        prior = self.prior(image, illum, reflectance)
        enhanced_illum = image * (prior / prior.mean(dim=(2, 3), keepdim=True).clamp(min=1e-2))
        fused = self.fuse(torch.cat([enhanced_illum, reflectance], dim=1))
        chroma = self.chroma_branch(image)
        return (fused + 0.1 * chroma).clamp(0.0, 1.0), illumination
