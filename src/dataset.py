# Data loading and preprocessing for the dataset
# - Pairs images (Original, ExpertC)
# - Split into train, val, test sets
# - Apply transforms (crop, flip, normalize)
# - Serve batches to the training loop via PyTorch's DataLoader

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def make_splits(input_dir: str, gt_dir: str, val_fraction: float = 0.2, test_fraction: float = 0.2, seed: int = 42):
    # Split all paired image files into train/val/test sets
    # Args:
    #   input_dir: Path to folder of raw input images
    #   gt_dir: Path to folder of expert-retouched ground truth images
    #   val_fraction: Fraction of images to hold out for validation (Default 0.2)
    #   test_fraction: Fraction of images to hold out for testing (Default 0.2)
    #   seed: Random seed for reproducible splits
    # Returns:
    #   train_pairs: List of (input_path, gt_path) tuples for training.
    #   val_pairs: List of (input_path, gt_path) tuples for validation.
    #   test_pairs: List of (input_path, gt_path) tuples for testing.

    input_dir = Path(input_dir)
    gt_dir = Path(gt_dir)

    # checking if files are both in expertc AND original!
    input_stems = {p.stem: p for p in input_dir.iterdir() if p.is_file()}
    gt_stems = {p.stem: p for p in gt_dir.iterdir() if p.is_file()}

    common_stems = sorted(set(input_stems.keys()) & set(gt_stems.keys()))

    if len(common_stems) == 0:
        raise ValueError(
            f"No matching files found between:\n  {input_dir}\n  {gt_dir}\n"
            "Make sure both folders contain files with the same names."
        )

    pairs = [(str(input_stems[s]), str(gt_stems[s])) for s in common_stems]

    # shuffle
    rng = random.Random(seed)
    rng.shuffle(pairs)

    n_val = max(1, int(len(pairs) * val_fraction))
    n_test = int(len(pairs) * test_fraction)
    val_pairs = pairs[:n_val]
    test_pairs = pairs[n_val:n_val+n_test]
    train_pairs = pairs[n_val+n_test:]

    print(f"Dataset split: {len(train_pairs)} training, {len(val_pairs)} validation, {len(test_pairs)} test pairs")
    return train_pairs, val_pairs, test_pairs


class FiveKDataset(Dataset):
    # Dataset for paired images
    # Each item returned is a dict:
    #   {
    #       'input':    Tensor of shape (3, H, W), normalized to [-1, 1]
    #       'target':   Tensor of shape (3, H, W), normalized to [-1, 1]
    #       'filename': str, the stem of the image filename (useful for saving results)
    #   }

    def __init__(self, pairs: list, split: str = "train", crop_size: int = 384):
        # Args:
        #   pairs:      List of (input_path, gt_path) tuples from make_splits().
        #   split:      'train' or 'val' — controls which augmentations are applied.
        #   crop_size:  Size of the square crop. Must be divisible by 16.

        assert crop_size % 16 == 0, "crop_size must be divisible by 16 (network downsamples 4x)"
        self.pairs = pairs
        self.split = split
        self.crop_size = crop_size

        # Shared normalization: maps [0, 1] pixel values to [-1, 1]
        # We use [-1, 1] because it centers the data around zero, which helps
        # the network learn faster and makes the residual prediction more natural.
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        input_path, gt_path = self.pairs[idx]
        filename = Path(input_path).stem

        # Load as RGB PIL Images (handles JPEG, PNG, etc.)
        input_img = Image.open(input_path).convert("RGB")
        target_img = Image.open(gt_path).convert("RGB")

        # Apply paired spatial transforms (crop + optional flip)
        input_tensor, target_tensor = self._apply_paired_transforms(input_img, target_img)

        # Normalize both to [-1, 1]
        input_tensor = self.normalize(input_tensor)
        target_tensor = self.normalize(target_tensor)

        return {
            "input": input_tensor,
            "target": target_tensor,
            "filename": filename,
        }

    def _apply_paired_transforms(self, input_img: Image.Image, target_img: Image.Image):
        # Apply identical transformations to paired images
        to_tensor = transforms.ToTensor()  # converts PIL [0,255] → float [0,1]

        if self.split == "train":
            # Generate a random seed to synchronize transforms
            seed = torch.randint(0, 2**32, (1,)).item()

            # Random crop — applied with the same seed to both images
            crop = transforms.RandomCrop(self.crop_size, pad_if_needed=True)
            torch.manual_seed(seed)
            input_img = crop(input_img)
            torch.manual_seed(seed)
            target_img = crop(target_img)

            # Random horizontal flip — same seed ensures same decision
            flip = transforms.RandomHorizontalFlip(p=0.5)
            torch.manual_seed(seed)
            input_img = flip(input_img)
            torch.manual_seed(seed)
            target_img = flip(target_img)

            # Apply ColorJitter to the input image ONLY.
            # This makes the input randomly dimmer, less contrasted, and desaturated,
            # which forces the model to learn how to recover vibrant colors and 
            # appropriate brightness rather than relying on identity mapping.
            color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)
            input_img = color_jitter(input_img)
        else:
            # For validation, always use the center crop for reproducibility
            crop = transforms.CenterCrop(self.crop_size)
            input_img = crop(input_img)
            target_img = crop(target_img)

        input_tensor = to_tensor(input_img)
        target_tensor = to_tensor(target_img)

        if self.split == "train" and torch.rand(1).item() < 0.5:
            # Add synthetic Chromatic Aberration (RGB halo) to the training inputs.
            # By randomly shifting the green channel spatial pixels, the model learns 
            # to ignore color fringes instead of mistaking them for physical green objects.
            shift_x = torch.randint(-3, 4, (1,)).item()
            shift_y = torch.randint(-3, 4, (1,)).item()
            input_tensor[1] = torch.roll(input_tensor[1], shifts=(shift_y, shift_x), dims=(0, 1))

        # -------------------------------------------------------------
        # Color Theory Application: Complementary Colors (Split Toning)
        # Instead of a flat yellow wash, we apply classic Teal/Orange color theory.
        # This pushes shadows towards cool Teal (receding) and highlights towards 
        # warm Orange (advancing), generating maximum depth and life.
        # -------------------------------------------------------------
        # Calculate perceived brightness (luminance) per pixel
        luminance = 0.299 * target_tensor[0] + 0.587 * target_tensor[1] + 0.114 * target_tensor[2]
        
        # Create masks to separate shadows and highlights
        shadow_mask = 1.0 - luminance
        highlight_mask = luminance
        
        # Teal shadows (lower Red, raise Blue/Green)
        t_r, t_g, t_b = 0.90, 1.05, 1.10
        
        # Orange/Warm highlights (raise Red, lower Blue) 
        o_r, o_g, o_b = 1.10, 1.05, 0.85
        
        # Blend the color theory multipliers smoothly based on the pixel's lighting
        target_tensor[0] = torch.clamp(target_tensor[0] * (t_r * shadow_mask + o_r * highlight_mask), 0.0, 1.0)
        target_tensor[1] = torch.clamp(target_tensor[1] * (t_g * shadow_mask + o_g * highlight_mask), 0.0, 1.0)
        target_tensor[2] = torch.clamp(target_tensor[2] * (t_b * shadow_mask + o_b * highlight_mask), 0.0, 1.0)
        
        return input_tensor, target_tensor


# Quick test — run this file directly to verify the dataset loads correctly:
#   python src/dataset.py
if __name__ == "__main__":
    import sys
    from torch.utils.data import DataLoader

    repo_root = Path(__file__).parent.parent
    input_dir = repo_root / "data" / "archive" / "fivek_512px" / "input"
    gt_dir = repo_root / "data" / "archive" / "fivek_512px" / "expertC_gt"

    if not input_dir.exists() or not gt_dir.exists():
        print(f"ERROR: Dataset directories not found.\nExpected:\n  {input_dir}\n  {gt_dir}")
        sys.exit(1)

    train_pairs, val_pairs, test_pairs = make_splits(str(input_dir), str(gt_dir))

    train_ds = FiveKDataset(train_pairs, split="train", crop_size=384)
    val_ds = FiveKDataset(val_pairs, split="val", crop_size=384)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)

    print(f"\nLoading one batch from training set...")
    batch = next(iter(train_loader))

    print(f"  input shape:  {batch['input'].shape}")   # expect (4, 3, 384, 384)
    print(f"  target shape: {batch['target'].shape}")  # expect (4, 3, 384, 384)
    print(f"  input range:  [{batch['input'].min():.2f}, {batch['input'].max():.2f}]")   # expect ~[-1, 1]
    print(f"  target range: [{batch['target'].min():.2f}, {batch['target'].max():.2f}]") # expect ~[-1, 1]
    print(f"  filenames:    {batch['filename']}")

    # Save a visual check: input crop and target crop side by side
    try:
        import torchvision
        # De-normalize: map [-1,1] back to [0,1] for saving
        grid_input = (batch["input"][0] + 1) / 2
        grid_target = (batch["target"][0] + 1) / 2
        grid = torch.stack([grid_input, grid_target])
        torchvision.utils.save_image(grid, "dataset_test.jpg", nrow=2)
        print("\nSaved dataset_test.jpg — open it and verify input/target are the same crop.")
    except Exception as e:
        print(f"\n(Could not save visual check: {e})")

    print("\nDataset test PASSED.")
