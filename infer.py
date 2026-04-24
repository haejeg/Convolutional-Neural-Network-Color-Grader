"""
infer.py — Inference script for running the trained model on new images.

Loads a saved model checkpoint and applies the retouching network to the provided images. 
Handles required padding since the U-Net architecture requires spatial dimensions to be 
divisible by 16.

HOW TO USE:
  python infer.py photo.jpg
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))
from src.model import UNet
from src.utils import get_device, tensor_to_pil


def parse_args():
    parser = argparse.ArgumentParser(description="Run photo retouching inference")
    parser.add_argument("image_name", type=str, nargs="?", default=None,
                        help="Input image name (e.g., 'hi.jpg') in the 'Input' folder. Leave blank to process the whole folder.")
    parser.add_argument("--input", type=str, default=None,
                        help="Alternative way to provide input image name or path")
    parser.add_argument("--input_dir", type=str, default="Input",
                        help="Input directory used when image_name/--input is not an absolute path (default: 'Input')")
    parser.add_argument("--output", type=str, default="results",
                        help="Output image path or directory (defaults to 'results')")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pth",
                        help="Path to model checkpoint (default: checkpoints/best.pth)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override: 'cpu', 'cuda', or 'mps'")

    # Optional runtime grading preferences (applied after the model output)
    parser.add_argument("--warmth", type=float, default=0.0,
                        help="Warm/cool shift in [-1, 1]. Negative=cool, positive=warm.")
    parser.add_argument("--tint", type=float, default=0.0,
                        help="Green/magenta shift in [-1, 1]. Negative=green, positive=magenta.")
    parser.add_argument("--sat", type=float, default=0.0,
                        help="Saturation adjustment in [-1, 1].")
    parser.add_argument("--contrast", type=float, default=0.0,
                        help="Contrast adjustment in [-1, 1].")
    parser.add_argument("--exposure", type=float, default=0.0,
                        help="Exposure in approximate stops (e.g., -1.0 darker, +1.0 brighter).")
    parser.add_argument("--grade_linear", action="store_true",
                        help="Apply warmth/tint in linear RGB for more photographic behavior.")
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device) -> UNet:
    """Loads the model and checkpoint for inference."""
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Train the model first with: python src/train.py"
        )

    model = UNet(in_channels=3, out_channels=3, base_channels=64)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Set evaluation mode to disable operations like dropout and batch normalization updates
    model.eval()
    model.to(device)

    epoch = checkpoint.get("epoch", "?")
    val_loss = checkpoint.get("val_loss", float("nan"))
    print(f"Loaded checkpoint from epoch {epoch} (val_loss={val_loss:.4f})")
    return model


def pad_to_multiple(tensor: torch.Tensor, multiple: int = 16):
    """
    Pads tensor spatial dimensions to the nearest target multiple.
    This ensures compatibility with the U-Net architecture downsampling/upsampling factors.
    """
    _, _, h, w = tensor.shape
    
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    if pad_h > 0 or pad_w > 0:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

    return tensor, (pad_h, pad_w)


def preprocess_image(image_path: str, device: torch.device) -> tuple:
    """Loads image data and transforms it into the format expected by the model."""
    img = Image.open(image_path).convert("RGB")
    original_size = (img.height, img.width)

    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    tensor = normalize(to_tensor(img)).unsqueeze(0) 

    tensor, padding = pad_to_multiple(tensor, multiple=16)
    tensor = tensor.to(device)

    return tensor, original_size, padding


def postprocess_tensor(tensor: torch.Tensor, original_size: tuple) -> Image.Image:
    """Converts the model output tensor back into a PIL Image and crops padding."""
    tensor = tensor.squeeze(0) 

    h_orig, w_orig = original_size
    tensor = tensor[:, :h_orig, :w_orig]

    return tensor_to_pil(tensor)


def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0.04045, x / 12.92, torch.pow((x + 0.055) / 1.055, 2.4))


def _linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * torch.pow(x, 1.0 / 2.4) - 0.055)


def apply_runtime_grade(
    pred: torch.Tensor,
    *,
    warmth: float = 0.0,
    tint: float = 0.0,
    saturation: float = 0.0,
    contrast: float = 0.0,
    exposure: float = 0.0,
    grade_linear: bool = False,
) -> torch.Tensor:
    """
    Apply simple, user-controlled grading to a predicted tensor.
    Expects pred in [-1, 1] and returns a tensor in [-1, 1].
    """
    x = (pred + 1.0) / 2.0
    x = x.clamp(0.0, 1.0)

    # Exposure in "stops": multiply by 2^stops
    if exposure != 0.0:
        x = x * (2.0 ** float(exposure))

    # Warmth/tint (either in sRGB-ish or linear RGB)
    if any(v != 0.0 for v in (warmth, tint)):
        if grade_linear:
            x_lin = _srgb_to_linear(x)
            r, g, b = x_lin[:, 0:1], x_lin[:, 1:2], x_lin[:, 2:3]
            r = r * (1.0 + 0.10 * float(warmth))
            b = b * (1.0 - 0.10 * float(warmth))
            g = g * (1.0 - 0.10 * float(tint))
            x = _linear_to_srgb(torch.cat([r, g, b], dim=1).clamp(0.0, 1.0))
        else:
            r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
            r = r * (1.0 + 0.10 * float(warmth))
            b = b * (1.0 - 0.10 * float(warmth))
            g = g * (1.0 - 0.10 * float(tint))
            x = torch.cat([r, g, b], dim=1)

    # Contrast around mid-gray
    if contrast != 0.0:
        c = 1.0 + 0.50 * float(contrast)
        x = (x - 0.5) * c + 0.5

    # Saturation: lerp between grayscale and original
    if saturation != 0.0:
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        s = 1.0 + 0.75 * float(saturation)
        x = gray + (x - gray) * s

    x = x.clamp(0.0, 1.0)
    return x * 2.0 - 1.0


@torch.no_grad() 
def retouch_image(model: UNet, image_path: str, output_path: str, device: torch.device, args):
    """Performs inference on a single image and saves the result."""
    print(f"  Processing: {Path(image_path).name}")

    tensor, original_size, padding = preprocess_image(image_path, device)
    pred = model(tensor)
    pred = apply_runtime_grade(
        pred,
        warmth=args.warmth,
        tint=args.tint,
        saturation=args.sat,
        contrast=args.contrast,
        exposure=args.exposure,
        grade_linear=args.grade_linear,
    )
    result_img = postprocess_tensor(pred, original_size)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result_img.save(output_path)
    print(f"  Saved:      {output_path}")


def main():
    args = parse_args()

    if args.device:
        device = torch.device(args.device)
        print(f"Using device: {device} (from --device flag)")
    else:
        device = get_device()

    model = load_model(args.checkpoint, device)

    input_given = args.image_name or args.input
    if input_given:
        input_path = Path(input_given)
        input_dir = Path(args.input_dir)
        if not input_path.exists() and (input_dir / input_given).exists():
            input_path = input_dir / input_given
        elif not input_path.exists() and not input_path.is_absolute():
            input_path = input_dir / input_given
    else:
        input_path = Path(args.input_dir)

    output_path = Path(args.output)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    if input_path.is_dir():
        image_files = [p for p in sorted(input_path.iterdir())
                       if p.suffix.lower() in image_extensions]

        if not image_files:
            print(f"No image files found in: {input_path}")
            sys.exit(1)

        output_path.mkdir(parents=True, exist_ok=True)
        print(f"\nProcessing {len(image_files)} images from {input_path}...\n")

        for img_path in image_files:
            out_file = output_path / img_path.name
            retouch_image(model, str(img_path), str(out_file), device, args)

        print(f"\nDone. {len(image_files)} images saved to {output_path}/")

    elif input_path.is_file():
        if input_path.suffix.lower() not in image_extensions:
            print(f"Unsupported file type: {input_path.suffix}")
            print(f"Supported: {', '.join(image_extensions)}")
            sys.exit(1)

        if output_path.is_dir() or not output_path.suffix:
            output_path = output_path / input_path.name

        print(f"\nRetouching {input_path}...\n")
        retouch_image(model, str(input_path), str(output_path), device, args)
        print("\nDone.")
    else:
        print(f"Input not found: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
