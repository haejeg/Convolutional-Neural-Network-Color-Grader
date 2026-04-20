"""
utils.py — Shared helper functions used across the project.

Includes:
  - Converting tensors back to PIL images for saving
  - Saving side-by-side comparison grids for visual inspection
  - Counting model parameters
  - Setting random seeds for reproducible experiments
"""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.utils import make_grid


def set_seed(seed: int = 42):
    """
    Set all random seeds for reproducibility across Python, NumPy, and PyTorch.
    Call this at the start of training to ensure consistent results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a single image tensor in [-1, 1] to a PIL Image in [0, 255].

    Args:
        tensor: Shape (3, H, W), values in [-1, 1]

    Returns:
        PIL Image in RGB mode.
    """
    # De-normalize from [-1, 1] to [0, 1]
    tensor = (tensor.detach().cpu() + 1.0) / 2.0
    # Clamp to valid range (in case of slight numerical overshoot)
    tensor = tensor.clamp(0.0, 1.0)
    # Convert to uint8 [0, 255] NumPy array and then to PIL
    array = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(array)


def save_comparison_grid(
    input_t: torch.Tensor,
    pred_t: torch.Tensor,
    target_t: torch.Tensor,
    save_path: str,
    num_images: int = 4,
):
    """
    Save a side-by-side grid of [Input | Prediction | Target | Difference].
    Very useful for visually tracking training progress.

    Args:
        input_t:    Batch of input tensors,      shape (B, 3, H, W), range [-1, 1]
        pred_t:     Batch of predicted tensors,  shape (B, 3, H, W), range [-1, 1]
        target_t:   Batch of ground truth,       shape (B, 3, H, W), range [-1, 1]
        save_path:  Where to save the image (e.g., 'results/epoch_10_samples.jpg')
        num_images: How many examples from the batch to include (default 4).
    """
    # Limit to available batch size
    n = min(num_images, input_t.shape[0])

    # De-normalize to [0, 1] for display
    inp = (input_t[:n].detach().cpu() + 1.0) / 2.0
    pred = (pred_t[:n].detach().cpu() + 1.0) / 2.0
    tgt = (target_t[:n].detach().cpu() + 1.0) / 2.0

    # Compute absolute difference (amplified 3x for visibility)
    diff = (tgt - pred).abs().clamp(0.0, 1.0) * 3.0
    diff = diff.clamp(0.0, 1.0)

    # Interleave: for each example, show [input, pred, target, diff] in a row
    rows = []
    for i in range(n):
        rows.extend([inp[i], pred[i], tgt[i], diff[i]])

    # make_grid arranges tensors into a single image with 4 columns (one per type)
    grid = make_grid(rows, nrow=4, padding=4, pad_value=0.5)

    # Convert to PIL and save
    grid_np = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid_np).save(save_path)


def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> torch.device:
    """
    Return the best available device: MPS (Apple Silicon) > CUDA > CPU.
    Prints which device was selected so you always know what's running.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("Using device: CPU (training will be slow)")
    return device
