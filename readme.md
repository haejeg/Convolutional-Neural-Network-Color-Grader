## Convolutional-Neural-Network-Color-Corrector

This project trains a U-Net style retouching model on paired images (original → expert edited) and runs inference on new photos.

### Dataset
- **Original dataset**: `https://www.kaggle.com/datasets/weipengzhang/adobe-fivek/`
- **Dataset used here**: `https://www.kaggle.com/datasets/jesucristo/mit-5k-basic`

Expected folder structure (default):

```
Data/
  Original/   # input photos
  ExpertC/    # target retouched photos (paired by filename)
Input/        # photos you want to retouch at runtime (infer.py default)
checkpoints/
results/
```

### Install

```bash
pip install -r requirements.txt
```

### Train

```bash
python src/train.py --data_dir Data --epochs 100 --batch_size 8
```

Training outputs are written to a run folder (default: `experiments/run_YYYYMMDD_HHMMSS/`):
- `config.json` — the exact CLI args used
- `splits.json` — the exact train/val/test file lists
- `results/metrics.csv` — per-epoch metrics (includes PSNR, SSIM, and mean ΔE)
- `results/test_metrics.json` — final test metrics for the best checkpoint
- `checkpoints/best.pth`, `checkpoints/last.pth`

### Inference

Run on a single image (looks in `Input/` by default):

```bash
python infer.py photo.jpg --checkpoint checkpoints/best.pth --output results/photo.jpg
```

Process the whole input folder:

```bash
python infer.py --output results
```

### Runtime grading “preferences” (optional)

These are applied **after** the model output at runtime:

```bash
python infer.py photo.jpg --warmth 0.2 --tint 0.1 --sat 0.3 --contrast 0.1 --exposure 0.0
```

If you want warmth/tint to behave more like photo editors, enable linear mode:

```bash
python infer.py photo.jpg --warmth 0.2 --tint 0.1 --grade_linear
```

### Controlled experiments (variants)

You can run multiple controlled variants (e.g., with/without GAN) and save all results:

```bash
python src/run_variants.py --variants experiments/variants.json
```

This writes a summary CSV to `experiments/summary_*.csv` and each variant run to its own folder under `experiments/`.