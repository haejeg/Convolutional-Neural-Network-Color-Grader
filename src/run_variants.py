"""
run_variants.py — Controlled experiment runner.

Reads a JSON file describing multiple training variants and runs them sequentially,
writing each run's outputs into its own run directory under experiments/.

Example:
  python src/run_variants.py --variants experiments/variants.json
"""

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Run controlled training variants")
    p.add_argument("--variants", type=str, default="experiments/variants.json", help="Path to variants JSON file")
    p.add_argument("--root", type=str, default="experiments", help="Experiments root folder")
    p.add_argument("--python", type=str, default=sys.executable, help="Python executable to use")
    return p.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).parent.parent

    variants_path = repo_root / args.variants
    if not variants_path.exists():
        raise FileNotFoundError(f"Variants file not found: {variants_path}")

    variants = json.loads(variants_path.read_text(encoding="utf-8"))
    if not isinstance(variants, list) or not variants:
        raise ValueError("Variants JSON must be a non-empty list of objects.")

    exp_root = repo_root / args.root
    exp_root.mkdir(parents=True, exist_ok=True)

    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = exp_root / f"summary_{batch_id}.csv"

    with open(summary_path, "w", newline="", encoding="utf-8") as fsum:
        writer = csv.DictWriter(
            fsum,
            fieldnames=[
                "variant",
                "run_dir",
                "test_loss",
                "test_psnr",
                "test_ssim",
                "test_delta_e",
            ],
        )
        writer.writeheader()

        for v in variants:
            name = v.get("name")
            train_args = v.get("train_args", {})
            if not name or not isinstance(train_args, dict):
                raise ValueError("Each variant must have: {\"name\": str, \"train_args\": {..}}")

            run_dir = exp_root / f"{batch_id}_{name}"
            run_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                args.python,
                str(repo_root / "src" / "train.py"),
                "--run_dir",
                str(run_dir),
            ]
            for k, val in train_args.items():
                flag = f"--{k}"
                if isinstance(val, bool):
                    if val:
                        cmd.append(flag)
                else:
                    cmd.extend([flag, str(val)])

            print(f"\n=== Running variant: {name} ===")
            print(" ".join(cmd))
            subprocess.run(cmd, check=True, cwd=str(repo_root))

            test_path = run_dir / "results" / "test_metrics.json"
            if test_path.exists():
                test = json.loads(test_path.read_text(encoding="utf-8"))
            else:
                test = {}

            writer.writerow(
                {
                    "variant": name,
                    "run_dir": str(run_dir),
                    "test_loss": test.get("loss", ""),
                    "test_psnr": test.get("psnr", ""),
                    "test_ssim": test.get("ssim", ""),
                    "test_delta_e": test.get("delta_e", ""),
                }
            )

    print(f"\nDone. Summary written to: {summary_path}")


if __name__ == "__main__":
    main()

