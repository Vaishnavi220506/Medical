# Sample scans

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
| `patient3_two_scans/scan1` and `scan2_later` | The **same patient scanned twice**. The tumor grew from about 12 cm³ to 23 cm³ between the scans, so this is a real example for tracking over time. |

**How to try it:**
1. Drag **all four files of one folder** into the upload box together (the sequences are detected from the file names). Enter a patient label and click **Analyze**. The 3D model takes about 1 minute on a laptop CPU.
2. Open the **3D** tab to rotate the tumor and switch its parts on and off: necrotic core (red), edema (yellow) and active, enhancing tumor (blue).
3. For tracking over time, upload `patient3_two_scans/scan1` and then `scan2_later` with the **same patient label** and two different scan dates (for example 2026-01-01 and 2026-04-01; BraTS doesn't publish the real dates). Then open the **Timeline** tab.

**Source and license:** BraTS 2023 Glioma (ASNR-MICCAI-BraTS2023-GLI), via https://huggingface.co/datasets/MedOtter/brats2023-gli-dataset (CC-BY-4.0). Please cite: U. Baid et al., "The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification", arXiv:2107.02314; B. H. Menze et al., IEEE TMI 34(10), 2015; S. Bakas et al., Scientific Data 4, 170117 (2017). The scans are de-identified and skull-stripped as distributed by the challenge.
