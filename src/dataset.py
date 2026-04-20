"""
dataset.py — Data loading and preprocessing for the MIT-Adobe FiveK dataset.

This module handles:
- Finding paired (raw input, expert-retouched) image files
- Splitting them into training and validation sets
- Applying the correct image transforms (crop, flip, normalize)
- Serving batches to the training loop via PyTorch's DataLoader

The most important thing to get right here is PAIRED transforms: when we
randomly crop or flip the input image, we must apply the exact same crop/flip
to the ground-truth target. If we don't, the network will try to learn a
mapping between misaligned image pairs, which is impossible.
"""

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def make_splits(input_dir: str, gt_dir: str, val_fraction: float = 0.1, seed: int = 42):
    """
    Find all paired image files and split them into train/val sets.

    Args:
        input_dir:     Path to folder of raw input images.
        gt_dir:        Path to folder of expert-retouched ground truth images.
        val_fraction:  Fraction of images to hold out for validation (default 10%).
        seed:          Random seed for reproducible splits.

    Returns:
        train_pairs: List of (input_path, gt_path) tuples for training.
        val_pairs:   List of (input_path, gt_path) tuples for validation.
    """
    input_dir = Path(input_dir)
    gt_dir = Path(gt_dir)

    # Collect filenames that exist in BOTH directories (inner join on filename stem)
    input_stems = {p.stem: p for p in input_dir.iterdir() if p.is_file()}
    gt_stems = {p.stem: p for p in gt_dir.iterdir() if p.is_file()}

    common_stems = sorted(set(input_stems.keys()) & set(gt_stems.keys()))

    if len(common_stems) == 0:
        raise ValueError(
            f"No matching files found between:\n  {input_dir}\n  {gt_dir}\n"
            "Make sure both folders contain files with the same names."
        )

    pairs = [(str(input_stems[s]), str(gt_stems[s])) for s in common_stems]

    # Shuffle deterministically so the split is always the same
    rng = random.Random(seed)
    rng.shuffle(pairs)

    n_val = max(1, int(len(pairs) * val_fraction))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    print(f"Dataset split: {len(train_pairs)} training, {len(val_pairs)} validation pairs")
    return train_pairs, val_pairs


class FiveKDataset(Dataset):
    """
    PyTorch Dataset for paired (input, ground-truth) image retouching.

    Each item returned is a dict:
        {
            'input':    FloatTensor of shape (3, H, W), normalized to [-1, 1]
            'target':   FloatTensor of shape (3, H, W), normalized to [-1, 1]
            'filename': str, the stem of the image filename (useful for saving results)
        }
    """

    def __init__(self, pairs: list, split: str = "train", crop_size: int = 384):
        """
        Args:
            pairs:      List of (input_path, gt_path) tuples from make_splits().
            split:      'train' or 'val' — controls which augmentations are applied.
            crop_size:  Size of the square crop. Must be divisible by 16.
        """
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
        """
        Apply identical spatial transforms to both images.

        The key trick: we set the same random seed before transforming each image,
        so that random operations (crop location, flip decision) are identical.
        Without this, the input and target would be different spatial regions,
        and training would fail silently.
        """
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
        else:
            # For validation, always use the center crop for reproducibility
            crop = transforms.CenterCrop(self.crop_size)
            input_img = crop(input_img)
            target_img = crop(target_img)

        return to_tensor(input_img), to_tensor(target_img)


# ---------------------------------------------------------------------------
# Quick test — run this file directly to verify the dataset loads correctly:
#   python src/dataset.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from torch.utils.data import DataLoader

    repo_root = Path(__file__).parent.parent
    input_dir = repo_root / "data" / "archive" / "fivek_512px" / "input"
    gt_dir = repo_root / "data" / "archive" / "fivek_512px" / "expertC_gt"

    if not input_dir.exists() or not gt_dir.exists():
        print(f"ERROR: Dataset directories not found.\nExpected:\n  {input_dir}\n  {gt_dir}")
        sys.exit(1)

    train_pairs, val_pairs = make_splits(str(input_dir), str(gt_dir))

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
