# L1 Loss & Perceptual Loss used here, a loss function used to generate high quality pioctures by comparing high level features
# extracted from a pre trained network (VGG16 here). Rather than comparing pixel by pixel like L1, compares feature maps.abs
# Total loss = 0.5 * L1_loss + 0.1 * perceptual_loss
# 0.1 to make output look more natural than it would be with just L1 loss.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG16_Weights


class PerceptualLoss(nn.Module):
    # Computes perceptual similarity through VGG16 layers
    # relu2_2 captures low-level detail (edges, fine textures)
    # relu3_3 captures mid-level structure (object parts, color blobs)
    # The VGG network is frozen — we never update its weights. We are only
    # using it as a fixed feature extractor!!

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
        # [-1, 1] -> [0, 1] -> ImageNet range
        x = (x + 1.0) / 2.0                       # [-1,1] → [0,1]
        x = (x - self.vgg_mean) / self.vgg_std     # [0,1] → ImageNet range
        return x

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Takes in pred and target images (outputs of network and ground truth)
        # Returns perceptual loss

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


def rgb_to_lab(image: torch.Tensor) -> torch.Tensor:
    # Convert [-1, 1] RGB to CIELAB space in a differentiable way.
    device = image.device
    
    # 1. [-1, 1] -> [0, 1]
    image = (image + 1.0) / 2.0
    image = torch.clamp(image, 0.0, 1.0)
    
    # 2. Inverse sRGB gamma correction
    mask = (image > 0.04045).type_as(image)
    # Add epsilon to prevent NaNs in pow gradient
    base = torch.clamp((image + 0.055) / 1.055, min=1e-5)
    image_linear = mask * torch.pow(base, 2.4) + (1.0 - mask) * image / 12.92
    
    # 3. sRGB to XYZ matrix
    matrix = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], device=device, dtype=image.dtype)
    
    B, C, H, W = image_linear.shape
    image_flat = image_linear.view(B, 3, H * W)
    xyz = torch.bmm(matrix.unsqueeze(0).expand(B, -1, -1), image_flat)
    xyz = xyz.view(B, 3, H, W)
    
    # 4. Normalize by D65 white point
    white_point = torch.tensor([0.95047, 1.00000, 1.08883], device=device, dtype=image.dtype).view(1, 3, 1, 1)
    xyz = xyz / white_point
    
    # 5. XYZ to LAB
    mask = (xyz > 0.008856).type_as(xyz)
    base_xyz = torch.clamp(xyz, min=1e-5)
    f_xyz = mask * torch.pow(base_xyz, 1.0/3.0) + (1.0 - mask) * (7.787 * xyz + 16.0 / 116.0)
    
    L = 116.0 * f_xyz[:, 1:2, :, :] - 16.0
    a = 500.0 * (f_xyz[:, 0:1, :, :] - f_xyz[:, 1:2, :, :])
    b = 200.0 * (f_xyz[:, 1:2, :, :] - f_xyz[:, 2:3, :, :])
    
    return torch.cat([L, a, b], dim=1)


def cielab_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_lab = rgb_to_lab(pred)
    target_lab = rgb_to_lab(target)
    # L1 distance in LAB space approximates Delta E
    return F.l1_loss(pred_lab, target_lab)


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    perceptual_loss_fn: PerceptualLoss,
    l1_weight: float = 0.5,
    cielab_weight: float = 0.5,
    perceptual_weight: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    # Compute the combined L1 + perceptual loss.

    # args:
    #    pred:               Predicted image, shape (B, 3, H, W), range [-1, 1]
    #    target:             Ground truth,   shape (B, 3, H, W), range [-1, 1]
    #    perceptual_loss_fn: An instance of PerceptualLoss.
    #    l1_weight:          Weight for the pixel-level L1 loss (default 1.0).
    #    perceptual_weight:  Weight for the perceptual loss (default 0.1).

    # Returns:
    #    total_loss: Scalar tensor to call .backward() on.
    #    components: Dict with individual loss values for logging.

    l1 = F.l1_loss(pred, target)
    cielab = cielab_loss(pred, target)
    perceptual = perceptual_loss_fn(pred, target)

    total = l1_weight * l1 + cielab_weight * cielab + perceptual_weight * perceptual

    return total, {
        "l1": l1.item(),
        "cielab": cielab.item(),
        "perceptual": perceptual.item(),
        "total": total.item()
    }


class GANLoss(nn.Module):
    # Gan Loss here (LSGAN or vanilla GAN)
    def __init__(self, use_lsgan: bool = True):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        # Create label tensors with the same size as the input prediction.
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def forward(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)


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
    print(f"CIELAB loss:     {components['cielab']:.4f}")
    print(f"Perceptual loss: {components['perceptual']:.4f}")
    print(f"Total loss:      {components['total']:.4f}")
    assert loss.ndim == 0, "Loss must be a scalar tensor"
    assert not torch.isnan(loss), "Loss must not be NaN"

    print("\nLoss test PASSED.")
