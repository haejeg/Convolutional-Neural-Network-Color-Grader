"""
metrics.py — Evaluation metrics for measuring retouching quality.

We use two standard image quality metrics:

  PSNR (Peak Signal-to-Noise Ratio):
    Measures the ratio between the maximum possible signal and the noise
    (error) between prediction and ground truth. Reported in decibels (dB).
    Higher is better. Typical range for retouching: 25–35 dB.
    Infinite PSNR means the images are identical.

  SSIM (Structural Similarity Index):
    Measures similarity in terms of luminance, contrast, and structure.
    Range: [0, 1]. Higher is better. Values above 0.85 are generally good.
    Unlike PSNR, SSIM correlates well with human perception of image quality.

IMPORTANT: Both metrics must be computed on images in [0, 1] pixel space,
not in our training range of [-1, 1]. Always de-normalize first.
"""

import torch
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert a tensor from training range [-1, 1] back to [0, 1].
    This is required before computing PSNR or SSIM.
    """
    return (tensor + 1.0) / 2.0


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute PSNR between prediction and ground truth.

    Args:
        pred:   Predicted image tensor, shape (B, 3, H, W), range [-1, 1]
        target: Ground truth tensor,   shape (B, 3, H, W), range [-1, 1]

    Returns:
        PSNR value in dB (scalar float). Higher is better.

    Note: We de-normalize both tensors to [0, 1] before computing.
    The data_range=1.0 argument tells torchmetrics that pixel values
    span [0, 1], which it needs to compute the "peak" in PSNR.
    """
    pred_01 = denormalize(pred)
    target_01 = denormalize(target)
    return peak_signal_noise_ratio(pred_01, target_01, data_range=1.0).item()


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute SSIM between prediction and ground truth.

    Args:
        pred:   Predicted image tensor, shape (B, 3, H, W), range [-1, 1]
        target: Ground truth tensor,   shape (B, 3, H, W), range [-1, 1]

    Returns:
        SSIM value in [0, 1] (scalar float). Higher is better.
    """
    pred_01 = denormalize(pred)
    target_01 = denormalize(target)
    return structural_similarity_index_measure(pred_01, target_01, data_range=1.0).item()


def evaluate_batch(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """
    Compute both PSNR and SSIM for a batch of images.

    Args:
        pred:   Shape (B, 3, H, W), range [-1, 1]
        target: Shape (B, 3, H, W), range [-1, 1]

    Returns:
        Dict with keys 'psnr' and 'ssim', each a scalar float.
    """
    return {
        "psnr": compute_psnr(pred, target),
        "ssim": compute_ssim(pred, target),
    }


# ---------------------------------------------------------------------------
# Quick test — run this file directly:
#   python src/metrics.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Test 1: identical images should give very high PSNR and SSIM ≈ 1.0
    identical = torch.zeros(2, 3, 64, 64)
    metrics_identical = evaluate_batch(identical, identical)
    print(f"Identical images → PSNR: {metrics_identical['psnr']:.2f} dB, SSIM: {metrics_identical['ssim']:.4f}")
    assert metrics_identical["ssim"] > 0.999, "SSIM for identical images should be ~1.0"

    # Test 2: completely different images should give low PSNR and SSIM
    pred = torch.randn(2, 3, 64, 64).clamp(-1, 1)
    target = torch.randn(2, 3, 64, 64).clamp(-1, 1)
    metrics_random = evaluate_batch(pred, target)
    print(f"Random images    → PSNR: {metrics_random['psnr']:.2f} dB, SSIM: {metrics_random['ssim']:.4f}")
    assert metrics_random["psnr"] < 20, "PSNR for random images should be low"

    print("\nMetrics test PASSED.")
