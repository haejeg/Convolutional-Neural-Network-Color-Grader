"""
train.py — Training loop for the U-Net photo retouching model.

HOW TO USE:
  # Smoke test first (2 epochs, small batch, no parallelism):
  python src/train.py --epochs 2 --batch_size 2 --num_workers 0

  # Full training run:
  python src/train.py --epochs 100 --batch_size 8 --num_workers 4

  # Resume from a checkpoint:
  python src/train.py --epochs 100 --resume checkpoints/last.pth

OUTPUTS:
  checkpoints/last.pth  — saved every epoch (overwritten); use for recovery
  checkpoints/best.pth  — saved only when val loss improves; use for inference
  results/epoch_N.jpg   — side-by-side visual samples saved every 5 epochs
  results/metrics.csv   — loss/PSNR/SSIM logged every epoch for plotting
"""

import argparse
import csv
import sys
from pathlib import Path

# Add project root to path so we can import from src/
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import FiveKDataset, make_splits
from src.losses import PerceptualLoss, combined_loss
from src.metrics import evaluate_batch
from src.model import UNet, count_parameters
from src.utils import get_device, save_comparison_grid, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train the U-Net retouching model")
    parser.add_argument("--data_dir", type=str,
                        default="Data",
                        help="Path to dataset directory (containing Original/ and ExpertC/)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Total number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size (reduce to 4 if you get out-of-memory errors)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the total number of image pairs to process (useful for quick testing)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for Adam optimizer")
    parser.add_argument("--crop_size", type=int, default=384,
                        help="Square crop size (must be divisible by 16)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker processes (use 0 on macOS if you see errors)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint file to resume training from")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--save_interval", type=int, default=5,
                        help="Save visual sample grid every N epochs")
    return parser.parse_args()


def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, best_val_loss, path: str):
    """Save training state so we can resume later or deploy the model."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
    }, path)


def load_checkpoint(path: str, model, optimizer, scheduler, device) -> tuple[int, float]:
    """
    Load a checkpoint and restore model/optimizer/scheduler state.
    Returns (start_epoch, best_val_loss) so training can resume correctly.
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    # Restore best_val_loss so we don't incorrectly overwrite best.pth on resume
    best_val_loss = checkpoint.get("best_val_loss", checkpoint["val_loss"])
    print(f"Resumed from checkpoint: epoch {checkpoint['epoch']}, best_val_loss={best_val_loss:.4f}")
    return start_epoch, best_val_loss


def train_one_epoch(model, loader, optimizer, perceptual_fn, device, epoch) -> dict:
    """
    Run one full pass through the training set.
    Returns averaged metrics over the epoch.
    """
    model.train()
    total_loss = total_l1 = total_perceptual = 0.0
    total_psnr = total_ssim = 0.0

    progress = tqdm(loader, desc=f"Epoch {epoch:03d} [train]", leave=False)
    for batch in progress:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)

        # Forward pass
        preds = model(inputs)

        # Compute loss
        loss, components = combined_loss(preds, targets, perceptual_fn)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate metrics (detach from graph before computing quality metrics)
        with torch.no_grad():
            metrics = evaluate_batch(preds.detach(), targets)
            total_loss += components["total"]
            total_l1 += components["l1"]
            total_perceptual += components["perceptual"]
            total_psnr += metrics["psnr"]
            total_ssim += metrics["ssim"]

        progress.set_postfix(loss=f"{components['total']:.4f}")

    n = len(loader)
    return {
        "loss": total_loss / n,
        "l1": total_l1 / n,
        "perceptual": total_perceptual / n,
        "psnr": total_psnr / n,
        "ssim": total_ssim / n,
    }


@torch.no_grad()
def validate(model, loader, perceptual_fn, device) -> dict:
    """
    Run one full pass through the validation set without updating weights.
    Returns averaged metrics and the last batch for visualization.
    """
    model.eval()
    total_loss = total_l1 = total_perceptual = 0.0
    total_psnr = total_ssim = 0.0
    last_batch = None

    progress = tqdm(loader, desc="           [val]  ", leave=False)
    for batch in progress:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)

        preds = model(inputs)
        loss, components = combined_loss(preds, targets, perceptual_fn)
        metrics = evaluate_batch(preds, targets)

        total_loss += components["total"]
        total_l1 += components["l1"]
        total_perceptual += components["perceptual"]
        total_psnr += metrics["psnr"]
        total_ssim += metrics["ssim"]
        last_batch = (inputs.cpu(), preds.cpu(), targets.cpu())

        progress.set_postfix(loss=f"{components['total']:.4f}")

    n = len(loader)
    return {
        "loss": total_loss / n,
        "l1": total_l1 / n,
        "perceptual": total_perceptual / n,
        "psnr": total_psnr / n,
        "ssim": total_ssim / n,
        "last_batch": last_batch,
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    repo_root = Path(__file__).parent.parent
    data_dir = repo_root / args.data_dir
    input_dir = data_dir / "Original"
    gt_dir = data_dir / "ExpertC"

    if not input_dir.exists() or not gt_dir.exists():
        print(f"ERROR: Dataset not found at {data_dir}")
        print("Expected subdirectories: Original/ and ExpertC/")
        sys.exit(1)

    train_pairs, val_pairs = make_splits(str(input_dir), str(gt_dir))

    if args.limit is not None:
        all_pairs = val_pairs + train_pairs
        if len(all_pairs) > args.limit:
            all_pairs = all_pairs[:args.limit]
            n_val = max(1, int(len(all_pairs) * 0.1))
            val_pairs = all_pairs[:n_val]
            train_pairs = all_pairs[n_val:]
            print(f"Limiting dataset to: {len(train_pairs)} training, {len(val_pairs)} validation pairs")

    train_ds = FiveKDataset(train_pairs, split="train", crop_size=args.crop_size)
    val_ds = FiveKDataset(val_pairs, split="val", crop_size=args.crop_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    # ------------------------------------------------------------------
    # Model, optimizer, scheduler, loss
    # ------------------------------------------------------------------
    model = UNet(in_channels=3, out_channels=3, base_channels=64).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # CosineAnnealingLR smoothly reduces the learning rate to near zero
    # over the full training run, which prevents oscillation late in training.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    perceptual_fn = PerceptualLoss(device=device)

    # ------------------------------------------------------------------
    # Resume from checkpoint (optional)
    # ------------------------------------------------------------------
    start_epoch = 1
    best_val_loss = float("inf")

    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(args.resume, model, optimizer, scheduler, device)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    checkpoints_dir = repo_root / "checkpoints"
    results_dir = repo_root / "results"
    checkpoints_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    # CSV log — append mode so resuming doesn't wipe previous runs
    csv_path = results_dir / "metrics.csv"
    csv_fields = ["epoch", "train_loss", "train_psnr", "train_ssim",
                  "val_loss", "val_psnr", "val_ssim", "lr"]
    csv_is_new = not csv_path.exists()

    print(f"\nStarting training for {args.epochs} epochs...\n")

    for epoch in range(start_epoch, args.epochs + 1):
        # Train
        train_metrics = train_one_epoch(model, train_loader, optimizer, perceptual_fn, device, epoch)

        # Validate
        val_metrics = validate(model, val_loader, perceptual_fn, device)

        # Advance learning rate schedule
        scheduler.step()

        # Log epoch summary to stdout
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train loss: {train_metrics['loss']:.4f} | "
            f"Val loss: {val_metrics['loss']:.4f} | "
            f"Val PSNR: {val_metrics['psnr']:.2f} dB | "
            f"Val SSIM: {val_metrics['ssim']:.4f} | "
            f"LR: {lr:.2e}"
        )

        # Append metrics to CSV (write header only on first row)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            if csv_is_new:
                writer.writeheader()
                csv_is_new = False
            writer.writerow({
                "epoch": epoch,
                "train_loss": f"{train_metrics['loss']:.6f}",
                "train_psnr": f"{train_metrics['psnr']:.4f}",
                "train_ssim": f"{train_metrics['ssim']:.4f}",
                "val_loss": f"{val_metrics['loss']:.6f}",
                "val_psnr": f"{val_metrics['psnr']:.4f}",
                "val_ssim": f"{val_metrics['ssim']:.4f}",
                "lr": f"{lr:.2e}",
            })

        # Always save the latest checkpoint (overwrite previous)
        save_checkpoint(model, optimizer, scheduler, epoch, val_metrics["loss"], best_val_loss,
                        str(checkpoints_dir / "last.pth"))

        # Save the best checkpoint when validation loss improves
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics["loss"], best_val_loss,
                            str(checkpoints_dir / "best.pth"))
            print(f"  ✓ New best model saved (val_loss={best_val_loss:.4f})")

        # Save visual comparison grid every N epochs
        if epoch % args.save_interval == 0 and val_metrics["last_batch"] is not None:
            inp_b, pred_b, tgt_b = val_metrics["last_batch"]
            save_path = str(results_dir / f"epoch_{epoch:03d}.jpg")
            save_comparison_grid(inp_b, pred_b, tgt_b, save_path)
            print(f"  Saved sample grid: {save_path}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {checkpoints_dir / 'best.pth'}")


if __name__ == "__main__":
    # On macOS, DataLoader with num_workers > 0 can deadlock with the default
    # 'fork' multiprocessing method. 'spawn' avoids this.
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
