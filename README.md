# Image K-Means Segmenter

This project implements an image segmentation tool using K-Means clustering. It segments an input image into color-based clusters and highlights the smallest cluster in green. The program also uses the Elbow Method to suggest an optimal number of clusters.

## Features

- Load and resize images.
- K-Means clustering with K-Means++ inspired initialization implemented from scratch.
- Highlights the smallest cluster (by pixel count) in green.
- Computes and plots inertia values for different cluster counts.
- Detects the elbow point to find the best number of clusters.
- Displays the segmented image for the chosen cluster count.

## Requirements

- Python 3.6+
- NumPy
- Pillow (PIL)
- Matplotlib

Install dependencies with:

```bash
pip install numpy pillow matplotlib
