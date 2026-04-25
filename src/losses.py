"""
losses.py — Loss functions for training the retouching network.

We use two losses combined:
  1. L1 Loss (pixel-level):
       Compares each pixel directly. L1 is preferred over MSE because
       MSE penalizes large errors very heavily, causing the network to
       "play it safe" by predicting blurry averages. L1 is more tolerant
       of outlier pixels and produces sharper outputs.

  2. Perceptual Loss (feature-level):
       Instead of comparing pixels, we pass both the prediction and target
       through a pre-trained VGG16 network and compare their intermediate
       feature representations. This forces the network to match high-level
       structure (textures, edges, color regions) in addition to pixel values.
       Even a small weight (0.1) makes outputs look noticeably more natural.

Combined: total_loss = 1.0 * L1 + 0.1 * perceptual
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG16_Weights


class PerceptualLoss(nn.Module):
    """
    Computes perceptual similarity using frozen VGG16 feature layers.

    We extract features at two intermediate layers:
      - relu2_2: captures low-level detail (edges, fine textures)
      - relu3_3: captures mid-level structure (object parts, color blobs)

    The VGG network is frozen — we never update its weights. We are only
    using it as a fixed feature extractor.

    IMPORTANT: VGG was trained on ImageNet images normalized with a specific
    mean and std. Our images are normalized to [-1, 1]. We must convert back
    to the VGG-expected range inside forward(), or the perceptual features
    will be computed on wrong-scale inputs and the loss will be meaningless.
    """

    # ImageNet normalization that VGG expects
    _VGG_MEAN = torch.tensor([0.485, 0.456, 0.406])
    _VGG_STD = torch.tensor([0.229, 0.224, 0.225])

    def __init__(self, device: torch.device):
        super().__init__()

        # Load pre-trained VGG16 (downloads weights once on first run)
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features

        # We only need the first 16 layers to reach relu3_3
        # VGG16 feature layer indices: relu2_2 = layer 9, relu3_3 = layer 16
        self.slice1 = nn.Sequential(*list(vgg.children())[:9])   # up to relu2_2
        self.slice2 = nn.Sequential(*list(vgg.children())[9:16]) # up to relu3_3

        # Freeze all VGG parameters — we never want to update them
        for param in self.parameters():
            param.requires_grad = False

        # Register mean/std as buffers so they move to the correct device
        # automatically when we call .to(device)
        self.register_buffer("vgg_mean", self._VGG_MEAN.view(1, 3, 1, 1))
        self.register_buffer("vgg_std", self._VGG_STD.view(1, 3, 1, 1))

        self.to(device)

    def _normalize_for_vgg(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert from our [-1, 1] normalization to VGG's ImageNet normalization.
        Steps:
          1. [-1, 1] → [0, 1]  (de-normalize our encoding)
          2. [0, 1] → ImageNet  (apply VGG's expected normalization)
        """
        x = (x + 1.0) / 2.0                       # [-1,1] → [0,1]
        x = (x - self.vgg_mean) / self.vgg_std     # [0,1] → ImageNet range
        return x

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   Predicted image tensor, shape (B, 3, H, W), range [-1, 1]
            target: Ground truth tensor,   shape (B, 3, H, W), range [-1, 1]

        Returns:
            Scalar perceptual loss (mean over batch and feature locations).
        """
        pred_vgg = self._normalize_for_vgg(pred)
        target_vgg = self._normalize_for_vgg(target)

        # Extract features at relu2_2
        pred_f1 = self.slice1(pred_vgg)
        target_f1 = self.slice1(target_vgg)

        # Extract features at relu3_3 (continuing from relu2_2 output)
        pred_f2 = self.slice2(pred_f1)
        target_f2 = self.slice2(target_f1)

        # L1 distance in feature space at both layers
        loss = F.l1_loss(pred_f1, target_f1) + F.l1_loss(pred_f2, target_f2)
        return loss


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    perceptual_loss_fn: PerceptualLoss,
    l1_weight: float = 1.0,
    perceptual_weight: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """
    Compute the combined L1 + perceptual loss.

    Args:
        pred:               Predicted image, shape (B, 3, H, W), range [-1, 1]
        target:             Ground truth,   shape (B, 3, H, W), range [-1, 1]
        perceptual_loss_fn: An instance of PerceptualLoss.
        l1_weight:          Weight for the pixel-level L1 loss (default 1.0).
        perceptual_weight:  Weight for the perceptual loss (default 0.1).

    Returns:
        total_loss: Scalar tensor to call .backward() on.
        components: Dict with individual loss values for logging.
    """
    l1 = F.l1_loss(pred, target)
    perceptual = perceptual_loss_fn(pred, target)

    total = l1_weight * l1 + perceptual_weight * perceptual

    return total, {"l1": l1.item(), "perceptual": perceptual.item(), "total": total.item()}


# ---------------------------------------------------------------------------
# Quick test — run this file directly to verify the loss works:
#   python src/losses.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Using device: {device}")

    perceptual_fn = PerceptualLoss(device=device)

    # Two random "images" in [-1, 1] range
    pred = torch.randn(2, 3, 384, 384, device=device)
    target = torch.randn(2, 3, 384, 384, device=device)

    loss, components = combined_loss(pred, target, perceptual_fn)

    print(f"L1 loss:         {components['l1']:.4f}")
    print(f"Perceptual loss: {components['perceptual']:.4f}")
    print(f"Total loss:      {components['total']:.4f}")
    assert loss.ndim == 0, "Loss must be a scalar tensor"
    assert not torch.isnan(loss), "Loss must not be NaN"

    print("\nLoss test PASSED.")
