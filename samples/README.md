# Sample scans

**Easiest way to try them:** open the app and click one of the sample cards on the start page. It runs the full analysis on that scan; no upload needed. The cards are listed in `catalog.json`.

## `brain_mri_3d/`: real 3D brain MRI (BraTS 2023)

Real glioma patients from the BraTS 2023 training data. **None of OncoVision's models was trained on them**: they belong to the held-out test patients of the 3D model. Each folder has the four MRI sequences of one scan as NIfTI volumes:

| File ending | Sequence |
|---|---|
| `-t1c` | T1 after contrast (T1ce) |
| `-t1n` | T1 |
| `-t2f` | FLAIR |
| `-t2w` | T2 |

| Folder | What it shows |
|---|---|
| `patient1/` | Large tumor (about 135 cm³ including edema) |
| `patient2/` | Smaller tumor (about 34 cm³) |
| `patient3_two_scans/scan1` and `scan2_later` | The **same patient scanned twice**. The total tumor stayed about the same (about 104 cm³ both times) while the active, enhancing part shrank from about 44 to 36 cm³. This makes a real example for tracking over time; OncoVision reports it as stable disease (it measures 102 → 108 cm³). |

**How to try it:**
1. Drag **all four files of one folder** into the upload box together (the sequences are detected from the file names). Enter a patient label and click **Analyze**. The 3D model takes about 1 minute on a laptop CPU.
2. Open the **3D** tab to rotate the tumor and switch its parts on and off: necrotic core (red), edema (yellow) and active, enhancing tumor (blue).
3. For tracking over time, upload `patient3_two_scans/scan1` and then `scan2_later` with the **same patient label** and two different scan dates (for example 2026-01-01 and 2026-04-01; BraTS doesn't publish the real dates). Then open the **Timeline** tab.

**Source and license:** BraTS 2023 Glioma (ASNR-MICCAI-BraTS2023-GLI), via https://huggingface.co/datasets/MedOtter/brats2023-gli-dataset (CC-BY-4.0). Please cite: U. Baid et al., "The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification", arXiv:2107.02314; B. H. Menze et al., IEEE TMI 34(10), 2015; S. Bakas et al., Scientific Data 4, 170117 (2017). The scans are de-identified and skull-stripped as distributed by the challenge.

## `breast_mri_3d/`: real 3D breast MRI (MAMA-MIA)

Two breast cancer patients from the MAMA-MIA collection (first contrast-enhanced DCE-MRI phase). **Neither was used to train OncoVision's breast MRI models**: they are held-out test patients.

| File | What it shows |
|---|---|
| `patient1_ISPY2_943633.nii.gz` | A tumor of about 7 cm³ (the model's outline matches the expert one with Dice 0.93) |
| `patient2_ISPY2_986734.nii.gz` | A larger tumor of about 28 cm³ (Dice 0.85) |
| `patient3_DUKE_592_both_breasts.nii.gz` | A scan of **both breasts** (Duke cohort), with the cancer in the right breast (about 27 cm³). The 3D view shows it inside the drawn breast anatomy; the model also marks a false alarm in the left breast. This is the "Breast MRI, 3D" sample card on the start page. |

**How to try it:** drag one file into the upload box and click **Analyze** (scan type: Auto-detect). It is recognised as breast MRI and the 3D model outlines the tumor; open the **3D** tab and hover over it. The ISPY2 scans show only one breast, so the app does not say left or right for them (and the 3D view shows the scanned area instead of the drawn anatomy).

**Source and license:** MAMA-MIA (L. Garrucho et al., Scientific Data 12, 453, 2025), via https://huggingface.co/datasets/YongchengYAO/MAMA-MIA-Lite, **CC BY-NC 4.0: non-commercial use only**. The original ISPY2 data come from The Cancer Imaging Archive.

## `lung_ct_3d/`: real chest CT (LIDC-IDRI)

| File | What it shows |
|---|---|
| `LIDC-IDRI-0289_ct.nii.gz` | A chest CT with one nodule of about 2.7 cm in the right lung. The patient is one of the held-out test patients of the lung model; the app finds the nodule with no false alarms (size-based Lung-RADS 4B). |
| `LIDC-IDRI-0289_nodule_mask.nii.gz` | The radiologists' nodule outline, for comparison (not needed by the app). |

**How to try it:** drag `LIDC-IDRI-0289_ct.nii.gz` into the upload box and click **Analyze**. It takes 1–3 minutes on a laptop CPU, seconds on a GPU. The lung model is experimental: it finds about half of the nodules radiologists mark.

**Source and license:** LIDC-IDRI (S. G. Armato III et al., Medical Physics 38(2), 2011), The Cancer Imaging Archive, CC BY 3.0, via the NIfTI conversion at https://huggingface.co/datasets/totalol/lungct (orientation fixed with `scripts/make_lung_samples.py`).

## `images/`: single 2D images

| File | What it shows | Source |
|---|---|---|
| `brain_glioma.jpg`, `brain_meningioma.jpg`, `brain_pituitary.jpg` | Brain MRI slices with a glioma, a meningioma and a pituitary tumor | Test split of the Brain Tumor MRI Dataset (Masoud Nickparvar), CC BY 4.0 mirror |
| `breast_us_malignant.png`, `breast_us_benign.png` | Breast ultrasound with a biopsy-proven malignant and benign lesion | Held-out test split of BUS-BRA (W. Gómez-Flores et al., Medical Physics 2024), CC BY 4.0 |

All of them come from the test splits used to measure the models.
