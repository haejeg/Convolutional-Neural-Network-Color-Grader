"""
infer.py — Run the trained U-Net model on new images.

HOW TO USE:
  # Retouch a single image:
  python infer.py --input path/to/photo.jpg --output results/retouched.jpg

  # Retouch all images in a folder:
  python infer.py --input path/to/folder/ --output results/

  # Use a specific checkpoint (default: checkpoints/best.pth):
  python infer.py --input photo.jpg --output out.jpg --checkpoint checkpoints/last.pth

The model was trained on 384x384 crops, but at inference we want to process
the full image at its original resolution. We do this by:
  1. Padding the image dimensions up to the nearest multiple of 16
     (because the network downsamples 4 times, each by factor 2: 2^4 = 16)
  2. Running the model
  3. Cropping the output back to the original image size
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
    parser.add_argument("--input", type=str, required=True,
                        help="Input image path or directory of images")
    parser.add_argument("--output", type=str, required=True,
                        help="Output image path or directory")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pth",
                        help="Path to model checkpoint (default: checkpoints/best.pth)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override: 'cpu', 'cuda', or 'mps'")
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device) -> UNet:
    """
    Load the U-Net model from a saved checkpoint.

    Args:
        checkpoint_path: Path to the .pth file saved during training.
        device:          Device to load the model onto.

    Returns:
        The model in evaluation mode, ready for inference.
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Train the model first with: python src/train.py"
        )

    model = UNet(in_channels=3, out_channels=3, base_channels=64)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # eval() switches BatchNorm and Dropout to inference mode.
    # Forgetting this causes BatchNorm to use batch statistics instead of
    # the running statistics computed during training, giving wrong results.
    model.eval()
    model.to(device)

    epoch = checkpoint.get("epoch", "?")
    val_loss = checkpoint.get("val_loss", float("nan"))
    print(f"Loaded checkpoint from epoch {epoch} (val_loss={val_loss:.4f})")
    return model


def pad_to_multiple(tensor: torch.Tensor, multiple: int = 16):
    """
    Pad a tensor's spatial dimensions to the nearest multiple of `multiple`.

    The U-Net downsamples 4 times (each by 2), so input spatial dimensions
    must be divisible by 2^4 = 16. Images that aren't exactly divisible
    will fail unless we pad them first.

    Returns:
        padded_tensor: The padded tensor.
        (pad_h, pad_w): How much padding was added, so we can crop it off after.
    """
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    if pad_h > 0 or pad_w > 0:
        # Pad on the right and bottom sides (easier to crop off cleanly)
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

    return tensor, (pad_h, pad_w)


def preprocess_image(image_path: str, device: torch.device) -> tuple:
    """
    Load and preprocess an image for model inference.

    Returns:
        tensor:        Shape (1, 3, H_padded, W_padded), normalized to [-1, 1]
        original_size: (H_original, W_original) for cropping output back
        padding:       (pad_h, pad_w) added during padding
    """
    img = Image.open(image_path).convert("RGB")
    original_size = (img.height, img.width)

    # Convert PIL → float tensor [0, 1] → normalize to [-1, 1]
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    tensor = normalize(to_tensor(img)).unsqueeze(0)  # add batch dimension

    # Pad spatial dims to multiples of 16
    tensor, padding = pad_to_multiple(tensor, multiple=16)
    tensor = tensor.to(device)

    return tensor, original_size, padding


def postprocess_tensor(tensor: torch.Tensor, original_size: tuple) -> Image.Image:
    """
    Convert model output back to a PIL Image at the original resolution.

    Args:
        tensor:        Shape (1, 3, H_padded, W_padded), range [-1, 1]
        original_size: (H_original, W_original) — used to crop off padding

    Returns:
        PIL Image in RGB mode.
    """
    tensor = tensor.squeeze(0)  # remove batch dimension → (3, H_padded, W_padded)

    # Crop back to original size — padding was added on right/bottom so this is clean
    h_orig, w_orig = original_size
    tensor = tensor[:, :h_orig, :w_orig]

    return tensor_to_pil(tensor)


@torch.no_grad()
def retouch_image(model: UNet, image_path: str, output_path: str, device: torch.device):
    """
    Load one image, run the model, and save the retouched result.

    Args:
        model:       Trained U-Net in eval mode.
        image_path:  Path to the input image.
        output_path: Where to save the retouched output.
        device:      Torch device.
    """
    print(f"  Processing: {Path(image_path).name}")

    tensor, original_size, padding = preprocess_image(image_path, device)

    # Run the model — no_grad() is set by the calling context (the decorator above)
    pred = model(tensor)

    result_img = postprocess_tensor(pred, original_size)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result_img.save(output_path)
    print(f"  Saved:      {output_path}")


def main():
    args = parse_args()

    # Determine device
    if args.device:
        device = torch.device(args.device)
        print(f"Using device: {device} (from --device flag)")
    else:
        device = get_device()

    model = load_model(args.checkpoint, device)

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Supported image formats
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    if input_path.is_dir():
        # Process all images in the directory
        image_files = [p for p in sorted(input_path.iterdir())
                       if p.suffix.lower() in image_extensions]

        if not image_files:
            print(f"No image files found in: {input_path}")
            sys.exit(1)

        output_path.mkdir(parents=True, exist_ok=True)
        print(f"\nProcessing {len(image_files)} images from {input_path}...\n")

        for img_path in image_files:
            out_file = output_path / img_path.name
            retouch_image(model, str(img_path), str(out_file), device)

        print(f"\nDone. {len(image_files)} images saved to {output_path}/")

    elif input_path.is_file():
        # Process a single image
        if input_path.suffix.lower() not in image_extensions:
            print(f"Unsupported file type: {input_path.suffix}")
            print(f"Supported: {', '.join(image_extensions)}")
            sys.exit(1)

        # If output is a directory, use the input filename inside it
        if output_path.is_dir() or not output_path.suffix:
            output_path = output_path / input_path.name

        print(f"\nRetouching {input_path}...\n")
        retouch_image(model, str(input_path), str(output_path), device)
        print("\nDone.")
    else:
        print(f"Input not found: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
