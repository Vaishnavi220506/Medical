---
title: About OncoVision's AI results
source: OncoVision documentation
url: https://github.com/Vaishnavi220506/Medical
---
## What the models do
OncoVision runs a scan-type router, a tumor classifier and a tumor segmenter. The classifier predicts the most likely category from the training classes only (for brain MRI: no tumor, glioma, meningioma, pituitary). It cannot recognise tumor types it was not trained on, such as metastases or lymphoma, and will map them to the closest known class.

## Confidence and uncertainty
Confidence is the model's predicted probability for its top class. Uncertainty is estimated with Monte-Carlo dropout: the model is run several times with random dropout, and disagreement between runs indicates uncertainty. High confidence does not guarantee correctness, especially for scans unlike the training data (different scanners, protocols or patient groups).

## Heatmaps
Grad-CAM heatmaps show which image regions most influenced the classifier's decision. They explain the model, not the disease: a heatmap on the tumor is reassuring, but a heatmap elsewhere suggests the model may be relying on irrelevant features.

## Measurements
Volumes and diameters are computed from the segmentation. When the scan's pixel size is unknown (for example plain JPG/PNG images), sizes are estimated from typical anatomy and may be off by 20% or more.

## Demo mode
Without trained weights, a classical image-processing detector is used. It highlights unusually bright or dark regions and cannot identify tumor types. Its results are for demonstrating the software only.
