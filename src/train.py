"""
train.py — Training loop for the U-Net photo retouching model.

Runs the training, validation, and testing phases. Integrates perceptual and adversarial losses 
to train the model to map raw inputs to expert retouched ground truth images.

HOW TO USE:
  # Quick test run:
  python src/train.py --epochs 2 --batch_size 2 --num_workers 0

  # Training Run with limited dataset:
  python src/train.py --epochs 100 --batch_size 8 --num_workers 4 --limit 100
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import FiveKDataset, make_splits
from src.losses import PerceptualLoss, combined_loss, GANLoss
from src.metrics import evaluate_batch
from src.model import UNet, count_parameters, PatchDiscriminator
from src.utils import get_device, save_comparison_grid, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train the U-Net retouching model")
    parser.add_argument("--data_dir", type=str,
                        default="Data",
                        help="Path to dataset directory (containing Original/ and ExpertC/)")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Directory to store outputs (config, splits, metrics, checkpoints). "
                             "If omitted, uses experiments/run_YYYYMMDD_HHMMSS")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Total number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training and validation loaders")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the total number of images (useful for testing)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for the optimizer")
    parser.add_argument("--crop_size", type=int, default=384,
                        help="Square crop dimension for training")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of background processes for data loading")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint file to resume training from")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for deterministic behavior")
    parser.add_argument("--save_interval", type=int, default=5,
                        help="Interval in epochs between saving visual comparison grids")

    # Variant controls for controlled experiments
    parser.add_argument("--use_gan", action="store_true",
                        help="Enable adversarial (GAN) training. If not set, trains without GAN.")
    parser.add_argument("--gan_weight", type=float, default=0.1,
                        help="Weight for adversarial loss when --use_gan is enabled.")
    parser.add_argument("--l1_weight", type=float, default=0.5,
                        help="Weight for pixel L1 loss in combined content loss.")
    parser.add_argument("--cielab_weight", type=float, default=0.5,
                        help="Weight for CIELAB loss in combined content loss.")
    parser.add_argument("--perceptual_weight", type=float, default=0.1,
                        help="Weight for VGG perceptual loss in combined content loss.")
    return parser.parse_args()


def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, best_val_loss, path: str, discriminator=None, optimizer_D=None, scheduler_D=None):
    """Saves model state and optimizer/scheduler configuration to disk."""
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
    }
    
    if discriminator:
        state["discriminator_state_dict"] = discriminator.state_dict()
    if optimizer_D:
        state["optimizer_D_state_dict"] = optimizer_D.state_dict()
    if scheduler_D:
        state["scheduler_D_state_dict"] = scheduler_D.state_dict()
        
    torch.save(state, path)


def load_checkpoint(path: str, model, optimizer, scheduler, device, discriminator=None, optimizer_D=None, scheduler_D=None) -> tuple[int, float]:
    """Restores model state from a saved checkpoint file."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    if discriminator and "discriminator_state_dict" in checkpoint:
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
    if optimizer_D and "optimizer_D_state_dict" in checkpoint:
        optimizer_D.load_state_dict(checkpoint["optimizer_D_state_dict"])
    if scheduler_D and "scheduler_D_state_dict" in checkpoint:
        scheduler_D.load_state_dict(checkpoint["scheduler_D_state_dict"])
        
    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint.get("best_val_loss", checkpoint["val_loss"])
    print(f"Resumed from checkpoint: epoch {checkpoint['epoch']}, best_val_loss={best_val_loss:.4f}")
    return start_epoch, best_val_loss


def train_one_epoch(
    model,
    discriminator,
    loader,
    optimizer,
    optimizer_D,
    perceptual_fn,
    gan_loss_fn,
    device,
    epoch,
    *,
    l1_weight: float,
    cielab_weight: float,
    perceptual_weight: float,
    gan_weight: float,
) -> dict:
    """Executes a single training epoch."""
    model.train() 
    if discriminator is not None:
        discriminator.train()
    
    total_loss = total_l1 = total_cielab = total_perceptual = total_adv = total_D_loss = 0.0
    total_psnr = total_ssim = total_delta_e = 0.0

    progress = tqdm(loader, desc=f"Epoch {epoch:03d} [train]", leave=False)
    for batch in progress:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)

        preds = model(inputs)

        loss_content, components = combined_loss(
            preds,
            targets,
            perceptual_fn,
            l1_weight=l1_weight,
            cielab_weight=cielab_weight,
            perceptual_weight=perceptual_weight,
        )

        loss_G_adv = torch.tensor(0.0, device=device)
        loss_D = torch.tensor(0.0, device=device)

        if discriminator is not None:
            # Update Discriminator
            pred_real = discriminator(inputs, targets)
            loss_D_real = gan_loss_fn(pred_real, True)
            
            pred_fake = discriminator(inputs, preds.detach())
            loss_D_fake = gan_loss_fn(pred_fake, False)
            
            loss_D = (loss_D_real + loss_D_fake) * 0.5

            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            # Update Generator (adversarial component)
            pred_fake_for_G = discriminator(inputs, preds)
            loss_G_adv = gan_loss_fn(pred_fake_for_G, True)

        loss_G = loss_content + gan_weight * loss_G_adv

        optimizer.zero_grad()
        loss_G.backward()
        optimizer.step()

        # Track metrics
        with torch.no_grad():
            metrics = evaluate_batch(preds.detach(), targets)
            total_loss += loss_G.item()
            total_l1 += components["l1"]
            total_cielab += components["cielab"]
            total_perceptual += components["perceptual"]
            total_adv += loss_G_adv.item()
            total_D_loss += loss_D.item()
            total_psnr += metrics["psnr"]
            total_ssim += metrics["ssim"]
            total_delta_e += metrics["delta_e"]

        if discriminator is not None:
            progress.set_postfix(G_loss=f"{loss_G.item():.4f}", D_loss=f"{loss_D.item():.4f}")
        else:
            progress.set_postfix(G_loss=f"{loss_G.item():.4f}")

    n = len(loader)
    return {
        "loss": total_loss / n,
        "l1": total_l1 / n,
        "cielab": total_cielab / n,
        "perceptual": total_perceptual / n,
        "adv": total_adv / n,
        "d_loss": total_D_loss / n,
        "psnr": total_psnr / n,
        "ssim": total_ssim / n,
        "delta_e": total_delta_e / n,
    }


@torch.no_grad()
def validate(model, loader, perceptual_fn, device) -> dict:
    """Executes validation over the provided dataloader without updating model weights."""
    model.eval()
    
    total_loss = total_l1 = total_cielab = total_perceptual = 0.0
    total_psnr = total_ssim = total_delta_e = 0.0
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
        total_cielab += components["cielab"]
        total_perceptual += components["perceptual"]
        total_psnr += metrics["psnr"]
        total_ssim += metrics["ssim"]
        total_delta_e += metrics["delta_e"]
        
        last_batch = (inputs.cpu(), preds.cpu(), targets.cpu())

        progress.set_postfix(loss=f"{components['total']:.4f}")

    n = len(loader)
    return {
        "loss": total_loss / n,
        "l1": total_l1 / n,
        "cielab": total_cielab / n,
        "perceptual": total_perceptual / n,
        "psnr": total_psnr / n,
        "ssim": total_ssim / n,
        "delta_e": total_delta_e / n,
        "last_batch": last_batch,
    }


def plot_learning_curves(csv_path: str, save_path: str):
    """Renders training and validation metrics as a line graph and saves to disk."""
    epochs, train_loss, val_loss = [], [], []
    train_psnr, val_psnr = [], []
    train_ssim, val_ssim = [], []
    
    if not Path(csv_path).exists():
        return
        
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epochs.append(int(row["epoch"]))
                train_loss.append(float(row["train_loss"]))
                val_loss.append(float(row["val_loss"]))
                train_psnr.append(float(row["train_psnr"]))
                val_psnr.append(float(row["val_psnr"]))
                train_ssim.append(float(row["train_ssim"]))
                val_ssim.append(float(row["val_ssim"]))
            except (ValueError, KeyError):
                continue
            
    if not epochs:
        return
        
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot(epochs, train_loss, label='Train Loss', color='blue', linewidth=2)
    axes[0].plot(epochs, val_loss, label='Val Loss', color='orange', linewidth=2)
    axes[0].set_title('Loss (L1 + Perceptual)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    axes[1].plot(epochs, train_psnr, label='Train PSNR', color='green', linewidth=2)
    axes[1].plot(epochs, val_psnr, label='Val PSNR', color='red', linewidth=2)
    axes[1].set_title('PSNR (Higher is better)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('dB')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    axes[2].plot(epochs, train_ssim, label='Train SSIM', color='purple', linewidth=2)
    axes[2].plot(epochs, val_ssim, label='Val SSIM', color='brown', linewidth=2)
    axes[2].set_title('SSIM (Higher is better)')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Index')
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    repo_root = Path(__file__).parent.parent
    run_dir = Path(args.run_dir) if args.run_dir else (repo_root / "experiments" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = run_dir / "checkpoints"
    results_dir = run_dir / "results"
    checkpoints_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    # Save run configuration early for reproducibility
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    data_dir = repo_root / args.data_dir
    input_dir = data_dir / "Original"
    gt_dir = data_dir / "ExpertC"

    if not input_dir.exists() or not gt_dir.exists():
        print(f"ERROR: Dataset not found at {data_dir}")
        print("Expected subdirectories: Original/ and ExpertC/")
        sys.exit(1)

    train_pairs, val_pairs, test_pairs = make_splits(str(input_dir), str(gt_dir))

    if args.limit is not None:
        all_pairs = test_pairs + val_pairs + train_pairs
        if len(all_pairs) > args.limit:
            all_pairs = all_pairs[:args.limit]
            n_val = max(1, int(len(all_pairs) * 0.1))
            n_test = int(len(all_pairs) * 0.1)
            val_pairs = all_pairs[:n_val]
            test_pairs = all_pairs[n_val:n_val+n_test]
            train_pairs = all_pairs[n_val+n_test:]
            print(f"Limiting dataset to: {len(train_pairs)} training, {len(val_pairs)} validation, {len(test_pairs)} test pairs")

    train_ds = FiveKDataset(train_pairs, split="train", crop_size=args.crop_size)
    val_ds = FiveKDataset(val_pairs, split="val", crop_size=args.crop_size)
    test_ds = FiveKDataset(test_pairs, split="val", crop_size=args.crop_size)

    # Save split lists for reproducibility
    def _pairs_to_records(pairs):
        return [{"input": p[0], "target": p[1], "stem": Path(p[0]).stem} for p in pairs]

    with open(run_dir / "splits.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": args.seed,
                "data_dir": str(data_dir),
                "train": _pairs_to_records(train_pairs),
                "val": _pairs_to_records(val_pairs),
                "test": _pairs_to_records(test_pairs),
            },
            f,
            indent=2,
        )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    model = UNet(in_channels=3, out_channels=3, base_channels=64).to(device)
    discriminator = PatchDiscriminator().to(device) if args.use_gan else None
    print(f"Generator parameters: {count_parameters(model):,}")
    if discriminator is not None:
        print(f"Discriminator parameters: {count_parameters(discriminator):,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr) if discriminator is not None else None

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scheduler_D = optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=args.epochs) if optimizer_D is not None else None

    perceptual_fn = PerceptualLoss(device=device)
    gan_loss_fn = GANLoss(use_lsgan=True).to(device)

    start_epoch = 1
    best_val_loss = float("inf")

    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(args.resume, model, optimizer, scheduler, device, discriminator, optimizer_D, scheduler_D)

    csv_path = results_dir / "metrics.csv"
    csv_fields = [
        "epoch",
        "train_loss", "train_psnr", "train_ssim", "train_delta_e",
        "val_loss", "val_psnr", "val_ssim", "val_delta_e",
        "lr",
    ]
    csv_is_new = not csv_path.exists()

    print(f"\nStarting training for {args.epochs} epochs...\n")

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            discriminator,
            train_loader,
            optimizer,
            optimizer_D,
            perceptual_fn,
            gan_loss_fn,
            device,
            epoch,
            l1_weight=args.l1_weight,
            cielab_weight=args.cielab_weight,
            perceptual_weight=args.perceptual_weight,
            gan_weight=args.gan_weight,
        )

        val_metrics = validate(model, val_loader, perceptual_fn, device)

        scheduler.step()
        if scheduler_D is not None:
            scheduler_D.step()

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"G loss: {train_metrics['loss']:.4f} | "
            f"D loss: {train_metrics['d_loss']:.4f} | "
            f"Val loss: {val_metrics['loss']:.4f} | "
            f"Val PSNR: {val_metrics['psnr']:.2f} dB | "
            f"Val SSIM: {val_metrics['ssim']:.4f} | "
            f"LR: {lr:.2e}"
        )

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
                "train_delta_e": f"{train_metrics['delta_e']:.4f}",
                "val_loss": f"{val_metrics['loss']:.6f}",
                "val_psnr": f"{val_metrics['psnr']:.4f}",
                "val_ssim": f"{val_metrics['ssim']:.4f}",
                "val_delta_e": f"{val_metrics['delta_e']:.4f}",
                "lr": f"{lr:.2e}",
            })

        plot_learning_curves(str(csv_path), str(results_dir / "learning_curves.png"))

        save_checkpoint(model, optimizer, scheduler, epoch, val_metrics["loss"], best_val_loss,
                        str(checkpoints_dir / "last.pth"), discriminator, optimizer_D, scheduler_D)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics["loss"], best_val_loss,
                            str(checkpoints_dir / "best.pth"), discriminator, optimizer_D, scheduler_D)
            print(f"  ✓ New best model saved (val_loss={best_val_loss:.4f})")

        if epoch % args.save_interval == 0 and val_metrics["last_batch"] is not None:
            inp_b, pred_b, tgt_b = val_metrics["last_batch"]
            save_path = str(results_dir / f"epoch_{epoch:03d}.jpg")
            save_comparison_grid(inp_b, pred_b, tgt_b, save_path)
            print(f"  Saved sample grid: {save_path}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {checkpoints_dir / 'best.pth'}")

    print("\n--- Running Final Evaluation on Test Set ---")
    
    best_checkpoint_path = checkpoints_dir / "best.pth"
    if best_checkpoint_path.exists():
        checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Loaded best model weights for test evaluation.")

    test_metrics = validate(model, test_loader, perceptual_fn, device)
    print(
        f"Test Loss: {test_metrics['loss']:.4f} | "
        f"Test PSNR: {test_metrics['psnr']:.2f} dB | "
        f"Test SSIM: {test_metrics['ssim']:.4f} | "
        f"Test ΔE: {test_metrics['delta_e']:.3f}"
    )

    with open(results_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "loss": float(test_metrics["loss"]),
                "psnr": float(test_metrics["psnr"]),
                "ssim": float(test_metrics["ssim"]),
                "delta_e": float(test_metrics["delta_e"]),
            },
            f,
            indent=2,
        )

    test_samples_dir = results_dir / "test_samples"
    test_samples_dir.mkdir(exist_ok=True)
    print(f"\nSaving test predictions to {test_samples_dir} ...")

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Saving Test Samples", leave=False)):
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            preds = model(inputs)

            save_path = str(test_samples_dir / f"test_batch_{i+1:03d}.jpg")
            save_comparison_grid(inputs.cpu(), preds.cpu(), targets.cpu(), save_path)
            
    print("Test evaluation complete! All test images saved.")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
