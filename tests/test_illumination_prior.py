import torch

from src.illumination_prior import IlluminationEstimator, GuidedPrior, retinex_decompose


def test_illumination_range():
    model = IlluminationEstimator(base_width=8)
    out = model(torch.rand(1, 3, 64, 64))
    assert out.shape == (1, 1, 64, 64)
    assert float(out.min()) >= 0.05 and float(out.max()) <= 1.0


def test_retinex_roundtrip():
    image = torch.rand(1, 3, 32, 32)
    illum = torch.ones(1, 1, 32, 32) * 0.5
    refl, _ = retinex_decompose(image, illum)
    assert torch.allclose(refl * illum, image, atol=1e-4)


def test_prior_gate_shape():
    prior = GuidedPrior(gate_hidden=4)
    image = torch.rand(2, 3, 16, 16)
    illum = torch.rand(2, 1, 16, 16)
    refl, _ = retinex_decompose(image, illum)
    out = prior(image, illum, refl)
    assert out.shape == (2, 1, 16, 16)
