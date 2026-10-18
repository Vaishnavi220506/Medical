# Training guide

You don't need a GPU on your laptop. The usual workflow is: **train on a free cloud GPU, then download the checkpoints into `weights/` on your laptop.**

Free options:
- **Kaggle Notebooks**: about 30 GPU hours per week, and the datasets are one click away (*Add Input*). Recommended.
- **Google Colab**: free T4 GPU, but sessions disconnect after a few hours, so save checkpoints to Google Drive.

The times below are rough estimates for a T4 GPU.

## 1. Set up the notebook

In a Kaggle or Colab notebook with a GPU enabled:

```bash
!git clone https://github.com/Vaishnavi220506/Medical.git oncovision
%cd oncovision
!pip install -q -r requirements.txt     # torch is already installed on Kaggle/Colab
```

On Kaggle, add the datasets with **Add Input** (they appear under `/kaggle/input/...`). On Colab, upload your `kaggle.json` API token and download them:

```bash
!mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p data --unzip
!kaggle datasets download -d aryashah2k/breast-ultrasound-images-dataset -p data --unzip
!kaggle datasets download -d dschettler8845/brats-2021-task1 -p data --unzip
```

Check the folder names with `!ls data` and adjust the paths below.

## 2. Brain tumor classifier (about 15-25 minutes)

```bash
!python -m oncovision.training.train_classifier --profile brain_mri \
    --train-dir data/Training --test-dir data/Testing \
    --pretrained --epochs 20 --batch-size 64
```

`--pretrained` starts from ImageNet weights, which trains faster and generalizes better. Test accuracy above 95% is typical on this dataset. Note that this dataset is "easy", so real-world accuracy will be lower.

## 3. Brain tumor segmenter on BraTS (prep about 30-45 min, training several hours)

```bash
!python -m oncovision.training.train_segmenter --profile brain_mri --format brats \
    --data data/BraTS2021_Training_Data --prepared-dir data/brats_slices \
    --epochs 30 --batch-size 32
```

The first run converts the 3D volumes into 2D slices (`--prepared-dir`) once. Later runs reuse them. If your session time is limited:
- train on a subset first: `--max-items 20000`
- use fewer epochs (`--epochs 10` already gives usable results)
- use a smaller network on small GPUs: `--base 16 --batch-size 16`

Good 2D U-Nets reach a validation whole-tumor Dice of about 0.85-0.90.

## 4. Breast ultrasound (about 10 minutes each)

```bash
!python -m oncovision.training.train_classifier --profile breast_us \
    --train-dir data/Dataset_BUSI_with_GT --pretrained --epochs 30 --batch-size 32
!python -m oncovision.training.train_segmenter --profile breast_us --format busi \
    --data data/Dataset_BUSI_with_GT --epochs 60 --batch-size 16
```

BUSI is small, so use more epochs. Results vary between runs.

## 5. Lung CT and the router

See [DATASETS.md](DATASETS.md) for the folder layouts, then:

```bash
!python -m oncovision.training.train_classifier --profile lung_ct --train-dir "data/The IQ-OTHNCCD lung cancer dataset" --pretrained
!python -m oncovision.training.train_classifier --router --train-dir data/router --pretrained --epochs 10
```

## 6. Bring the weights to your laptop

```bash
!zip -r weights.zip weights
```

Download `weights.zip`, unzip it into the project folder so you have `weights/brain_mri/classifier.pt` and so on, and restart the app (or call `POST /api/models/reload`). The status bar switches from "Demo mode" to "Trained models loaded", and `GET /api/health` shows each model's metrics.

## Tips

- Every script prints validation metrics each epoch and keeps the **best** checkpoint.
- Metrics are saved inside the checkpoint (`meta.metrics`) so you can quote them in your project report.
- Reproducibility: seeds are fixed (`seed_everything(42)`), and splits are deterministic.
- For a stronger brain segmenter, consider training a 3D model with MONAI or nnU-Net and adapting `Engine.segment`. The 2D approach here is chosen so it trains on free GPUs.
