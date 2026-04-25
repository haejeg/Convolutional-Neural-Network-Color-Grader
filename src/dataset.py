# dataset.py - Data Loading and Preparation
# 
# This module handles loading the image pairs, creating train/validation/test splits,
# applying data augmentations, and formatting the data into PyTorch tensors.

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def make_splits(input_dir: str, gt_dir: str, val_fraction: float = 0.2, test_fraction: float = 0.2, seed: int = 42):
    """
    Creates randomized data splits.
    
    Pairs input images with their corresponding ground truth (expert retouched) images,
    then splits them into three sets:
    1. Training set: Used to update the model weights.
    2. Validation set: Used to evaluate the model during training.
    3. Test set: Held out for final evaluation after training is complete.
    """
    input_dir = Path(input_dir)
    gt_dir = Path(gt_dir)

    # Get file stems (filenames without extensions) to find pairs
    input_stems = {p.stem: p for p in input_dir.iterdir() if p.is_file()}
    gt_stems = {p.stem: p for p in gt_dir.iterdir() if p.is_file()}

    # Only keep files that exist in both directories
    common_stems = sorted(set(input_stems.keys()) & set(gt_stems.keys()))

    if len(common_stems) == 0:
        raise ValueError(
            f"No matching files found between:\n  {input_dir}\n  {gt_dir}\n"
            "Make sure both folders contain files with the same names."
        )

    # Create tuples of (input_path, ground_truth_path)
    pairs = [(str(input_stems[s]), str(gt_stems[s])) for s in common_stems]

    # Shuffle the dataset randomly
    rng = random.Random(seed)
    rng.shuffle(pairs)

    # Calculate split sizes
    n_val = max(1, int(len(pairs) * val_fraction))
    n_test = int(len(pairs) * test_fraction)
    
    val_pairs = pairs[:n_val]
    test_pairs = pairs[n_val:n_val+n_test]
    train_pairs = pairs[n_val+n_test:]

    print(f"Dataset split: {len(train_pairs)} training, {len(val_pairs)} validation, {len(test_pairs)} test pairs")
    return train_pairs, val_pairs, test_pairs


class FiveKDataset(Dataset):
    """
    PyTorch Dataset for paired image loading.
    
    Loads image pairs, applies synchronized spatial transformations, 
    and normalizes the pixel values for the neural network.
    """

    def __init__(self, pairs: list, split: str = "train", crop_size: int = 384):
        # The U-Net architecture downsamples by a factor of 2 four times (2^4 = 16),
        # so input dimensions must be divisible by 16.
        assert crop_size % 16 == 0, "crop_size must be divisible by 16"
        
        self.pairs = pairs
        self.split = split
        self.crop_size = crop_size

        # Normalize pixel values from [0, 1] to [-1, 1] to center the data around zero
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        input_path, gt_path = self.pairs[idx]
        filename = Path(input_path).stem

        # Load images and ensure they are in RGB format
        input_img = Image.open(input_path).convert("RGB")
        target_img = Image.open(gt_path).convert("RGB")

        # Apply spatial augmentations and convert to tensors
        input_tensor, target_tensor = self._apply_paired_transforms(input_img, target_img)

        # Apply [-1, 1] normalization
        input_tensor = self.normalize(input_tensor)
        target_tensor = self.normalize(target_tensor)

        return {
            "input": input_tensor,
            "target": target_tensor,
            "filename": filename,
        }

    def _apply_paired_transforms(self, input_img: Image.Image, target_img: Image.Image):
        """
        Applies data augmentations to prevent overfitting.
        Spatial augmentations (crop, flip) must be applied identically to both images.
        Color augmentations are applied selectively to improve robustness.
        """
        to_tensor = transforms.ToTensor() 

        if self.split == "train":
            # Generate a shared random seed to ensure both images receive the exact same crop
            seed = torch.randint(0, 2**32, (1,)).item()
            crop = transforms.RandomCrop(self.crop_size, pad_if_needed=True)
            
            torch.manual_seed(seed)
            input_img = crop(input_img)
            
            torch.manual_seed(seed)
            target_img = crop(target_img)

            # Apply identical random horizontal flips
            flip = transforms.RandomHorizontalFlip(p=0.5)
            
            torch.manual_seed(seed)
            input_img = flip(input_img)
            
            torch.manual_seed(seed)
            target_img = flip(target_img)

            # Apply random color jitter to the input image only.
            # This forces the model to learn color correction rather than an identity mapping.
            color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)
            input_img = color_jitter(input_img)
            
        else:
            # Use deterministic center crops for validation and testing
            crop = transforms.CenterCrop(self.crop_size)
            input_img = crop(input_img)
            target_img = crop(target_img)

        input_tensor = to_tensor(input_img)
        target_tensor = to_tensor(target_img)

        # --- Synthetic Chromatic Aberration ---
        if self.split == "train" and torch.rand(1).item() < 0.5:
            # Randomly shift the green channel by a few pixels on the input image.
            # This teaches the model to correct color fringing artifacts.
            shift_x = torch.randint(-3, 4, (1,)).item()
            shift_y = torch.randint(-3, 4, (1,)).item()
            input_tensor[1] = torch.roll(input_tensor[1], shifts=(shift_y, shift_x), dims=(0, 1))

        # --- Complementary Color Grading Adjustment ---
        # Apply a split toning effect to the ground truth targets to encourage cinematic outputs.
        # Shadows are pushed slightly cool (teal) and highlights slightly warm (orange).
        
        # Calculate pixel luminance to use as a mask
        luminance = 0.299 * target_tensor[0] + 0.587 * target_tensor[1] + 0.114 * target_tensor[2]
        
        shadow_mask = 1.0 - luminance
        highlight_mask = luminance
        
        # Color multipliers for shadows (teal) and highlights (orange)
        t_r, t_g, t_b = 0.90, 1.05, 1.10
        o_r, o_g, o_b = 1.10, 1.05, 0.85
        
        # Blend the adjustments into the target tensor
        target_tensor[0] = torch.clamp(target_tensor[0] * (t_r * shadow_mask + o_r * highlight_mask), 0.0, 1.0)
        target_tensor[1] = torch.clamp(target_tensor[1] * (t_g * shadow_mask + o_g * highlight_mask), 0.0, 1.0)
        target_tensor[2] = torch.clamp(target_tensor[2] * (t_b * shadow_mask + o_b * highlight_mask), 0.0, 1.0)
        
        return input_tensor, target_tensor


if __name__ == "__main__":
    # Test script for dataset loading
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
