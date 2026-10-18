# Training guide

This guide trains the models that go in `weights/`. You don't need a GPU on your laptop: the easiest route is the notebook below on a free cloud GPU. Everything also runs on a CPU, just slower (a few hours on a 4-core CPU).

## Option A: one-click notebook (recommended)

1. Open [`notebooks/train_oncovision.ipynb`](../notebooks/train_oncovision.ipynb) on **Kaggle** (*New Notebook → File → Import Notebook*, then *Settings → Accelerator → GPU T4* and *Internet → On*) or **Google Colab** (*Runtime → Change runtime type → GPU*).
2. *Run all.* It clones this repo, downloads the datasets from public Hugging Face mirrors (no Kaggle API key needed), trains the brain and breast models, and writes evaluation reports.
3. Download `oncovision_weights.zip`, unzip it in your OncoVision folder, and restart the app.

With a T4 GPU and the notebook's default epochs this takes well under an hour in total.

## Option B: command line

### Get the data

```bash
pip install huggingface_hub pyarrow
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("BJyotibrat/Masoud-Nickparvar-Brain-Tumor-MRI-Dataset", repo_type="dataset", local_dir="data/brain")
snapshot_download("theshoaibme/panorgan-busi-breast-ultrasound", repo_type="dataset", local_dir="data/busi")
EOF
python scripts/prepare_figshare.py --download --raw data/figshare_raw --out data/figshare
```

For lung CT, download LIDC-IDRI volumes with nodule masks (see [DATASETS.md](DATASETS.md)) and convert them:

```bash
python scripts/prepare_lidc.py --raw data/lidc --out data/lidc_2d
```

### Train

```bash
# Brain: tumor type (4 classes) and tumor outline
python -m oncovision.training.train_classifier --profile brain_mri --train-dir data/brain/Training --test-dir data/brain/Testing --pretrained --epochs 15
python -m oncovision.training.train_segmenter  --profile brain_mri --format maskfolder --data data/figshare/train/images --masks data/figshare/train/masks --epochs 40

# Breast ultrasound
python -m oncovision.training.train_classifier --profile breast_us --train-dir data/busi --pretrained --epochs 30
python -m oncovision.training.train_segmenter  --profile breast_us --format busi --data data/busi --epochs 60

# Lung CT
python -m oncovision.training.train_classifier --profile lung_ct --train-dir data/lidc_2d/cls --pretrained --epochs 15
python -m oncovision.training.train_segmenter  --profile lung_ct --format maskfolder --data data/lidc_2d/seg/images --masks data/lidc_2d/seg/masks --epochs 40

# Scan-type router: one folder per type (brain_mri, breast_us, lung_ct, other), see DATASETS.md
python -m oncovision.training.train_classifier --router --train-dir data/router --pretrained --epochs 10
```

Useful options: `--batch-size`, `--lr`, `--workers`, `--max-items N` (quick experiments), `--base 16` (smaller U-Net for small GPUs), `--out path.pt`.

### BraTS sub-region segmenter (optional)

The Figshare segmenter outlines the **whole tumor** on single images (JPG/PNG/DICOM slices). To separate **necrotic core, edema and enhancing tumor** on multi-sequence MRI (FLAIR, T1, T1ce, T2 NIfTI), train on BraTS instead (about 13 GB, see [DATASETS.md](DATASETS.md)):

```bash
python -m oncovision.training.train_segmenter --profile brain_mri --format brats --data data/BraTS2021_Training_Data --prepared-dir data/brats_slices --epochs 20
```

The first run converts the 3D volumes into 2D slices once. On small GPUs use `--base 16 --batch-size 8`; with limited time use `--max-items 20000`. Good 2D U-Nets reach a whole-tumor Dice of about 0.85-0.90.

## Evaluate and make report figures

```bash
python -m oncovision.training.evaluate classifier --profile brain_mri --data data/brain/Testing
python -m oncovision.training.evaluate segmenter  --profile brain_mri --format maskfolder --data data/figshare/test/images --masks data/figshare/test/masks
python -m oncovision.training.evaluate segmenter  --profile breast_us --format busi --data data/busi --validation-only
```

Each run writes `reports/<profile>_<kind>/` with `metrics.json` plus figures for a project report: a confusion matrix and Grad-CAM examples for classifiers, and image / ground truth / prediction panels for segmenters.

## How the training is set up

- **Patient-level splits.** Files named `<patient>__<slice>.png` (Figshare, LIDC, BraTS) are split by patient, so neighbouring slices of one person never end up in both training and validation. Without this, scores look far better than they really are.
- **The best epoch is kept.** Each script evaluates every epoch and saves the best checkpoint. Its metrics are stored inside the checkpoint and shown by `GET /api/health`.
- **Same preprocessing as the app.** Training uses the same normalization, orientation and CT windowing (`oncovision.imaging`) as inference.
- **Augmentation**: random rotation, scaling, flips, gamma, contrast and noise. Multi-sequence BraTS models also get *sequence dropout*, so they still work when only one sequence is uploaded.
- **Losses**: class-weighted cross-entropy with label smoothing (classifiers), cross-entropy + Dice (segmenters).
- **Reproducible**: fixed seeds and deterministic splits.
