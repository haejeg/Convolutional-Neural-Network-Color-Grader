"""
metrics.py — Evaluation metrics for measuring model performance.

Includes PSNR and SSIM calculations using torchmetrics. Both metrics require
data in the [0, 1] range rather than the [-1, 1] training normalization.
"""

import torch
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Converts a tensor from the network's [-1, 1] training range back to [0, 1]
    for accurate metric calculation.
    """
    return (tensor + 1.0) / 2.0


def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    """Convert sRGB [0,1] to linear RGB [0,1]."""
    return torch.where(x <= 0.04045, x / 12.92, torch.pow((x + 0.055) / 1.055, 2.4))


def _rgb_to_lab_01(image_01: torch.Tensor) -> torch.Tensor:
    """
    Differentiable conversion from RGB in [0,1] to CIELAB.

    Output channels: L in [0..100] approximately, a/b roughly [-128..127] range.
    """
    device = image_01.device
    image_01 = torch.clamp(image_01, 0.0, 1.0)

    # sRGB -> linear RGB
    rgb_linear = _srgb_to_linear(image_01)

    # linear sRGB -> XYZ
    matrix = torch.tensor(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        device=device,
        dtype=image_01.dtype,
    )

    b, c, h, w = rgb_linear.shape
    rgb_flat = rgb_linear.view(b, 3, h * w)
    xyz = torch.bmm(matrix.unsqueeze(0).expand(b, -1, -1), rgb_flat)
    xyz = xyz.view(b, 3, h, w)

    # D65 white point normalization
    white_point = torch.tensor([0.95047, 1.00000, 1.08883], device=device, dtype=image_01.dtype).view(1, 3, 1, 1)
    xyz = xyz / white_point

    # XYZ -> Lab
    eps = 0.008856
    kappa = 7.787
    mask = (xyz > eps).type_as(xyz)
    f_xyz = mask * torch.pow(torch.clamp(xyz, min=1e-5), 1.0 / 3.0) + (1.0 - mask) * (kappa * xyz + 16.0 / 116.0)

    L = 116.0 * f_xyz[:, 1:2, :, :] - 16.0
    a = 500.0 * (f_xyz[:, 0:1, :, :] - f_xyz[:, 1:2, :, :])
    bch = 200.0 * (f_xyz[:, 1:2, :, :] - f_xyz[:, 2:3, :, :])
    return torch.cat([L, a, bch], dim=1)


def compute_delta_e(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Mean ΔE (CIE76) in LAB space. Lower is better.

    pred/target are expected in [-1,1].
    """
    pred_01 = denormalize(pred)
    target_01 = denormalize(target)
    pred_lab = _rgb_to_lab_01(pred_01)
    target_lab = _rgb_to_lab_01(target_01)
    delta = pred_lab - target_lab
    delta_e = torch.sqrt(torch.sum(delta * delta, dim=1) + 1e-12)  # (B,H,W)
    return delta_e.mean().item()


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).
    Higher values represent lower error between the prediction and target.
    Typical values range from 25 to 35 dB.
    """
    pred_01 = denormalize(pred)
    target_01 = denormalize(target)
    
    return peak_signal_noise_ratio(pred_01, target_01, data_range=1.0).item()


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Computes the Structural Similarity Index Measure (SSIM).
    SSIM evaluates luminance, contrast, and structural information, providing a 
    metric that generally correlates better with human visual perception than PSNR.
    Values range from [0, 1], where 1.0 represents perfect structural similarity.
    """
    pred_01 = denormalize(pred)
    target_01 = denormalize(target)
    
    return structural_similarity_index_measure(pred_01, target_01, data_range=1.0).item()


def evaluate_batch(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """
    Calculates both PSNR and SSIM for a batch of images and returns the values in a dictionary.
    """
    return {
        "psnr": compute_psnr(pred, target),
        "ssim": compute_ssim(pred, target),
        "delta_e": compute_delta_e(pred, target),
    }


if __name__ == "__main__":
    identical = torch.zeros(2, 3, 64, 64)
    metrics_identical = evaluate_batch(identical, identical)
    print(f"Identical images → PSNR: {metrics_identical['psnr']:.2f} dB, SSIM: {metrics_identical['ssim']:.4f}")
    assert metrics_identical["ssim"] > 0.999, "SSIM for identical images should be ~1.0"

    pred = torch.randn(2, 3, 64, 64).clamp(-1, 1)
    target = torch.randn(2, 3, 64, 64).clamp(-1, 1)
    metrics_random = evaluate_batch(pred, target)
    print(f"Random images    → PSNR: {metrics_random['psnr']:.2f} dB, SSIM: {metrics_random['ssim']:.4f}")
    assert metrics_random["psnr"] < 20, "PSNR for random images should be low"

    print("\nMetrics test PASSED.")
