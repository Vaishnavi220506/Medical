# Datasets

OncoVision is trained on public datasets. You download them yourself; most need a free account. **Check each dataset's license and citation requirements on its page before use**, and cite the dataset papers in any report or publication.

The training scripts read each dataset exactly as it is laid out after download. Folder names like `notumor`, `Bengin cases` or `Malignant cases` are mapped to OncoVision's class names automatically.

## Brain MRI

### Classifier: Brain Tumor MRI Dataset (Kaggle)
- Link: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
- About 7,000 2D MRI images in 4 classes: glioma, meningioma, pituitary, no tumor. It mixes axial, coronal and sagittal slices and different sequences.
- Layout:
  ```
  brain-tumor-mri-dataset/
    Training/{glioma,meningioma,notumor,pituitary}/*.jpg
    Testing/{glioma,meningioma,notumor,pituitary}/*.jpg
  ```

### Segmenter: BraTS 2021 (RSNA-ASNR-MICCAI Brain Tumor Segmentation)
- Official: https://www.synapse.org/brats (registration required). Kaggle mirror of the training set: https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1
- 1,251 pre-operative glioma cases. Each has 4 co-registered MRI sequences (FLAIR, T1, T1ce, T2), 1 mm isotropic, skull-stripped, plus an expert segmentation: 1 = necrotic core, 2 = edema, 4 = enhancing tumor. OncoVision maps 4 to 3.
- Layout:
  ```
  BraTS2021_Training_Data/
    BraTS2021_00000/
      BraTS2021_00000_flair.nii.gz
      BraTS2021_00000_t1.nii.gz
      BraTS2021_00000_t1ce.nii.gz
      BraTS2021_00000_t2.nii.gz
      BraTS2021_00000_seg.nii.gz
    ...
  ```
- BraTS 2023 naming (`*-t1n`, `*-t1c`, `*-t2w`, `*-t2f`, `*-seg`) is also supported.
- Cite: Baid U. et al., "The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification", arXiv:2107.02314; Menze B. H. et al., IEEE TMI 2015.

## Breast ultrasound

### Classifier + segmenter: BUSI (Breast Ultrasound Images Dataset)
- Link: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset
- 780 images from 600 women: 133 normal, 437 benign, 210 malignant, each with a lesion mask.
- Layout:
  ```
  Dataset_BUSI_with_GT/
    benign/benign (1).png, benign (1)_mask.png, ...
    malignant/...
    normal/...
  ```
  Images with several lesions have extra `_mask_1.png` files, which are merged automatically.
- Cite: Al-Dhabyani W. et al., "Dataset of breast ultrasound images", Data in Brief 28 (2020) 104863.

## Lung CT

### Classifier: IQ-OTH/NCCD lung cancer dataset
- Link: https://www.kaggle.com/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset
- About 1,100 CT slices in 3 classes: normal, benign, malignant.
- Layout: `The IQ-OTHNCCD lung cancer dataset/{Normal cases,Bengin cases,Malignant cases}/*.jpg` (the "Bengin" spelling is the dataset's own).

### Segmenter: any image + mask folder
- LIDC-IDRI (https://www.cancerimagingarchive.net/collection/lidc-idri/) has radiologist nodule outlines for 1,018 CT scans. Export 2D slices and masks (for example with the `pylidc` package) into:
  ```
  lung/images/<name>.png
  lung/masks/<name>.png     # 0 = background, 255 = nodule
  ```
  then train with `--format maskfolder --data lung/images --masks lung/masks`.

## Scan-type router

Build a folder with a few hundred images of each scan type, taken from the datasets above, plus an `other` folder with images OncoVision should reject (for example chest X-rays or knee MRI):

```
data/router/
  brain_mri/   (from the Kaggle brain dataset)
  breast_us/   (from BUSI, image files only, not masks)
  lung_ct/     (from IQ-OTH/NCCD)
  other/
```

Without a trained router, OncoVision guesses the scan type from DICOM metadata and image appearance, and you can always pick the type manually.
