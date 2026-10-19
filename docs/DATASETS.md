# Datasets

OncoVision's models are trained on public datasets. The Hugging Face mirrors below download without an account, which is what the training notebook uses. **Check each dataset's license and cite the original papers** in any report or publication. All the mirrors used are CC-BY-4.0, except where noted.

The training scripts read each dataset as it is laid out after download. Folder names like `notumor`, `Bengin cases` or `Malignant cases` are mapped to OncoVision's class names automatically.

## Brain MRI

### Tumor type: Brain Tumor MRI Dataset (Masoud Nickparvar)
- Hugging Face mirror: https://huggingface.co/datasets/BJyotibrat/Masoud-Nickparvar-Brain-Tumor-MRI-Dataset (CC-BY-4.0). Original: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
- 7,200 2D MRI images in 4 classes (glioma, meningioma, pituitary, no tumor), with a fixed 5,600 / 1,600 train/test split. It mixes axial, coronal and sagittal views and several sequences. It combines the Figshare, SARTAJ and Br35H datasets.
- Layout: `Training/{glioma,meningioma,notumor,pituitary}/*.jpg` and `Testing/...`

### Tumor outline: Figshare brain tumor dataset (Cheng et al.)
- Hugging Face mirror: https://huggingface.co/datasets/MedOtter/Figshare_Brain_Tumor (CC-BY-4.0, parquet). Original: https://figshare.com/articles/dataset/brain_tumor_dataset/1512427
- 3,064 T1 contrast-enhanced slices from 233 patients (glioma, meningioma, pituitary), each with an expert tumor mask. Pixel spacing is 0.49 mm. Patient-level folds are included.
- Convert it with `python scripts/prepare_figshare.py --download --out data/figshare`, which writes `train/` and `test/` (patient-disjoint) image and mask folders.
- Cite: Cheng J. et al., "Enhanced Performance of Brain Tumor Classification via Tumor Region Augmentation and Partition", PLoS ONE 10(10), 2015.

### 3D tumor sub-regions: BraTS 2023 (used for the shipped 3D model)
- Hugging Face mirror: https://huggingface.co/datasets/MedOtter/brats2023-gli-dataset (CC-BY-4.0, no login). 1,251 glioma scans with FLAIR, T1, T1ce, T2 and expert labels (1 necrotic core, 2 edema, 3 enhancing tumor). Some patients have several time points (`...-000`, `...-001`); all splits are by patient.
- Used by `notebooks/train_3d_brats.ipynb` / `oncovision.training.train_segmenter3d`.
- Cite: Baid U. et al., arXiv:2107.02314 (2021); Menze B. H. et al., IEEE TMI 34(10), 2015; Bakas S. et al., Scientific Data 4, 170117 (2017).

### Tumor sub-regions (older): BraTS 2021
- Official: https://www.synapse.org/brats (registration required). Kaggle mirror: https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1
- 1,251 glioma cases, each with 4 co-registered sequences (FLAIR, T1, T1ce, T2) and expert labels: 1 necrotic core, 2 edema, 4 enhancing tumor. OncoVision maps 4 to 3.
- Layout: `BraTS2021_Training_Data/BraTS2021_00000/BraTS2021_00000_{flair,t1,t1ce,t2,seg}.nii.gz`. BraTS 2023 names (`-t1n`, `-t1c`, `-t2w`, `-t2f`) also work.
- Cite: Baid U. et al., arXiv:2107.02314 (2021); Menze B. H. et al., IEEE TMI 34(10), 2015.

## Breast ultrasound

### BUSI (Breast Ultrasound Images Dataset)
- Hugging Face mirror: https://huggingface.co/datasets/theshoaibme/panorgan-busi-breast-ultrasound (CC-BY-4.0). Original: https://scholar.cu.edu.eg/?q=afahmy/pages/dataset
- 780 images from 600 women (133 normal, 437 benign, 210 malignant), each with a lesion mask. It is used for both the classifier and the segmenter.
- Layout: `{benign,malignant,normal}/benign (1).png` with `benign (1)_mask.png`. Extra `_mask_1.png` files are merged.
- Cite: Al-Dhabyani W. et al., "Dataset of breast ultrasound images", Data in Brief 28, 104863 (2020).

### More breast ultrasound: BUS-BRA and BUS-UCLM (used for the shipped breast models)
- **BUS-BRA** (Gómez-Flores W. et al., Medical Physics 2024): 1,875 biopsy-proven images (1,268 benign, 607 malignant) from 1,064 patients, with lesion masks. https://zenodo.org/records/8231412 (CC-BY-4.0).
- **BUS-UCLM** (Vallez N. et al., Scientific Data 2025): 683 images from 38 patients, 419 of them normal, with lesion masks. https://data.mendeley.com/datasets/7fvgj4jsp7 (CC-BY-4.0).
- `python scripts/prepare_breast.py --download --busi data/busi --out data/breast` combines them with BUSI into patient-disjoint train and test sets. The BUSI test images are the validation images of the earlier BUSI-only models, so old and new models can be compared on them.

## Breast MRI

### MAMA-MIA (3D breast tumors)
- 1,506 breast cancer DCE-MRI scans from four collections (Duke, ISPY1, ISPY2, NACT), each with an expert 3D outline of the primary tumor. Hugging Face redistribution: https://huggingface.co/datasets/YongchengYAO/MAMA-MIA-Lite. **License: CC BY-NC 4.0 (non-commercial use only).**
- Used by `notebooks/train_breast_mri3d.ipynb` / `oncovision.training.train_breast_mri3d` (3D model) and `export_slices` (2D model). All cases are cancers, so the models find and measure tumors but cannot tell benign from malignant.
- Cite: Garrucho L. et al., "A large-scale multicenter breast cancer DCE-MRI benchmark dataset with expert segmentations", Scientific Data 12, 453 (2025).

## Lung CT

### LIDC-IDRI (nodules)
- Original: https://www.cancerimagingarchive.net/collection/lidc-idri/ (CC BY 3.0). The bundled models used a subset (80 cases) of the NIfTI conversion at https://huggingface.co/datasets/totalol/lungct, which has nodule masks and a benign/malignant split by case. That mirror lists no license, so the original LIDC-IDRI terms apply.
- The full set (641 CTs) is used by `notebooks/train_lung3d.ipynb` / `oncovision.training.train_lung3d`, which streams the CTs one at a time.
- `scripts/prepare_lidc.py` converts the volumes into 2D slices with the lung window (-1000 to 400 HU), in standard radiological orientation (the mirror's NIfTI headers swap the x/y axes, which the script corrects). It writes:
  - `cls/{normal,benign,malignant}/<case>__<slice>.png`. Nodule slices take their case's label; "normal" slices are lung slices at least 10 slices away from any nodule.
  - `seg/images` + `seg/masks`: nodule slices plus some nodule-free slices.
- Note that "benign/malignant" here is a case-level label, not a biopsy result for each nodule, so treat the lung classifier as a rough guide.
- Cite: Armato S. G. III et al., Medical Physics 38(2), 915-931 (2011).

### IQ-OTH/NCCD (alternative lung classifier data)
- https://www.kaggle.com/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset
- About 1,100 CT slices: normal, benign, malignant. Layout: `{Normal cases,Bengin cases,Malignant cases}/*.jpg`.

## Scan-type router

The router learns to recognise the scan type and to reject unsupported images. Build one folder per type, for example:

```
data/router/
  brain_mri/   ~700 images (Kaggle brain + Figshare)
  breast_us/   ~600 BUSI images (not the masks)
  lung_ct/     ~600 LIDC slices
  other/       ~450 images the app should reject: chest X-rays
               (https://huggingface.co/datasets/hf-vision/chest-xray-pneumonia, CC-BY-4.0)
               and bone X-rays (https://huggingface.co/datasets/Mahadih534/x-ray_bone-fracture-Dataset)
```

Without a trained router, OncoVision falls back to DICOM metadata and simple image heuristics, and you can always pick the scan type manually.
