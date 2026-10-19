# Sample scans

## `brain_mri_3d/`: real 3D brain MRI (BraTS 2023)

Two glioma patients from the BraTS 2023 training data. Both were **held out**: no OncoVision model was trained on them. Each folder has the four MRI sequences of one patient as NIfTI volumes:

| File ending | Sequence |
|---|---|
| `-t1c` | T1 after contrast (T1ce) |
| `-t1n` | T1 |
| `-t2f` | FLAIR |
| `-t2w` | T2 |

**Try it:** in OncoVision, drag all four files of one patient into the upload box together (the sequences are detected from the file names), use the same patient label for both uploads, and click **Analyze**. Then open the **3D** tab to rotate the tumor with its sub-regions (necrotic core, edema, enhancing tumor) and scroll through the slices in the viewer.

`patient1` has a large tumor (about 135 cm³ including edema); `patient2` a smaller one (about 49 cm³).

**Source and license:** BraTS 2023 Glioma (ASNR-MICCAI-BraTS2023-GLI), via https://huggingface.co/datasets/MedOtter/brats2023-gli-dataset (CC-BY-4.0). Please cite: U. Baid et al., "The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification", arXiv:2107.02314; B. H. Menze et al., IEEE TMI 34(10), 2015; S. Bakas et al., Scientific Data 4, 170117 (2017). The scans are de-identified and skull-stripped as distributed by the challenge.
