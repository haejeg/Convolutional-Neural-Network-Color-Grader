"""
model.py — U-Net architecture for image-to-image photo retouching.

Architecture overview:
  The U-Net has an encoder (left side) that progressively shrinks the image
  while learning "what's in it", and a decoder (right side) that rebuilds
  the full-resolution image. Skip connections carry fine spatial detail
  (edges, textures) directly from encoder to decoder, preventing blur.

  Depth-4 design for our ~3000-sample dataset:
    Input (384x384) → Enc1 (384x384) → Enc2 (192x192) → Enc3 (96x96)
    → Enc4 (48x48) → Bottleneck (24x24)
    → Dec4 (48x48) → Dec3 (96x96) → Dec2 (192x192) → Dec1 (384x384)
    → Output (384x384)

  Note: Enc1 is a plain DoubleConv with no pooling, so spatial size stays
  at 384×384. Only Enc2–Enc4 and Bottleneck use Down (MaxPool + DoubleConv)
  to halve the spatial dimensions.

  Residual output:
    Instead of predicting the full retouched image, the network predicts
    a *correction delta* that gets added back to the raw input. This means
    at the start of training the network outputs ~zero correction (near
    identity), so losses start low and training is stable and fast.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Two back-to-back: Conv2d → BatchNorm → ReLU blocks.
    This is the core building block used everywhere in the U-Net.

    BatchNorm normalizes activations so that gradients flow smoothly
    during training — without it, deep networks are much harder to train.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """
    Encoder step: halve spatial size then apply DoubleConv.
    MaxPool2d picks the most active feature in each 2x2 window,
    effectively summarizing what's in that region.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    """
    Decoder step: double spatial size, concatenate skip connection, then DoubleConv.

    We use bilinear upsampling (not ConvTranspose2d) because bilinear
    interpolation avoids the "checkerboard artifact" pattern that
    ConvTranspose2d can produce when not carefully tuned.

    The skip connection: we concatenate the feature map from the matching
    encoder level. This doubles the channel count before DoubleConv
    reduces it back, which is why in_channels is the sum of both.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Upsample doubles spatial resolution; DoubleConv then blends the
        # upsampled features with the skip-connection features
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Handle edge case: if input spatial size isn't perfectly divisible,
        # the upsampled size may be off by 1 pixel. Pad to match skip size.
        if x.shape != skip.shape:
            x = F.pad(x, [0, skip.shape[3] - x.shape[3], 0, skip.shape[2] - x.shape[2]])

        # Concatenate along the channel dimension
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Full depth-4 U-Net for image-to-image retouching.

    Args:
        in_channels:   Number of input image channels (3 for RGB).
        out_channels:  Number of output image channels (3 for RGB).
        base_channels: Channel count at the first encoder level. Doubles at
                       each level down. Default 64 gives ~8M parameters.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_channels: int = 64):
        super().__init__()
        b = base_channels  # shorthand

        # Encoder (left side of U)
        self.enc1 = DoubleConv(in_channels, b)        # 384 → 384, 3→64 ch
        self.enc2 = Down(b, b * 2)                    # 384 → 192, 64→128 ch
        self.enc3 = Down(b * 2, b * 4)                # 192 → 96,  128→256 ch
        self.enc4 = Down(b * 4, b * 8)                # 96  → 48,  256→512 ch

        # Bottleneck (bottom of U) — deepest, most abstract representation
        self.bottleneck = Down(b * 8, b * 8)          # 48  → 24,  512→512 ch

        # Decoder (right side of U) — each Up takes upsampled + skip channels
        self.dec4 = Up(b * 8 + b * 8, b * 4)         # 512+512 → 256 ch
        self.dec3 = Up(b * 4 + b * 4, b * 2)         # 256+256 → 128 ch
        self.dec2 = Up(b * 2 + b * 2, b)             # 128+128 → 64 ch
        self.dec1 = Up(b + b, b // 2)                 # 64+64   → 32 ch

        # Final 1x1 convolution maps to output channels
        self.final_conv = nn.Conv2d(b // 2, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw_input = x  # save for residual addition at the end

        # Encoder — save each level's output as a skip connection
        s1 = self.enc1(x)          # skip from level 1
        s2 = self.enc2(s1)         # skip from level 2
        s3 = self.enc3(s2)         # skip from level 3
        s4 = self.enc4(s3)         # skip from level 4

        # Bottleneck
        neck = self.bottleneck(s4)

        # Decoder — each step receives the upsampled output + matching skip
        x = self.dec4(neck, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        # ---------------------------------------------------------
        # GLOBAL FILTER UPGRADE:
        # Instead of predicting a localized parameter for every pixel, we collapse 
        # the network's understanding into a single global set of parameters!
        # This acts EXACTLY like a Lightroom preset or color filter applied 
        # to the whole photo uniformly.
        # ---------------------------------------------------------
        x_global = torch.mean(x, dim=(2, 3), keepdim=True)  # Pool spatial dims -> shape (B, C, 1, 1)

        # Project to just 3 curve parameters (R, G, B) for the ENTIRE image
        global_curve_params = torch.tanh(self.final_conv(x_global))

        # Because `global_curve_params` is exactly the same for every pixel, 
        # it is physically and mathematically impossible for color to bleed 
        # across spatial boundaries! No more green trees bleeding into skies.
        x_01 = (raw_input + 1.0) / 2.0
        
        # Apply the RGB curves
        for _ in range(3):
            x_01 = x_01 + global_curve_params * x_01 * (1.0 - x_01)
            
        # Convert back
        output = torch.clamp((x_01 * 2.0) - 1.0, -1.0, 1.0)
        return output


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Quick test — run this file directly to verify the model works:
#   python src/model.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = UNet(in_channels=3, out_channels=3, base_channels=64)
    model.eval()

    # Create a fake batch: 1 image of shape (3, 384, 384), values in [-1, 1]
    dummy_input = torch.randn(1, 3, 384, 384)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Input shape:  {dummy_input.shape}")  # expect (1, 3, 384, 384)
    print(f"Output shape: {output.shape}")       # expect (1, 3, 384, 384)
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")  # expect within [-1, 1]
    print(f"Trainable parameters: {count_parameters(model):,}")        # expect ~8M

    # Verify shapes match (required for loss computation)
    assert output.shape == dummy_input.shape, "Output shape must match input shape!"
    print("\nModel test PASSED.")
