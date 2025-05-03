# Image K-Means Segmentation

This project implements image segmentation using the **K-means clustering algorithm** on image pixels. It incorporates important preprocessing steps such as Gaussian blurring and working in the perceptually uniform **CIELAB color space** to achieve better and more meaningful segmentation results.

---

## Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
  - [Requirements](#requirements)
  - [Installation](#installation)
- [Usage](#usage)
- [How it Works](#how-it-works)
- [Parameters](#parameters)
- [Output](#output)
- [Example](#example)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- Loads and resizes images for performance.
- Applies Gaussian blur to reduce noise.
- Converts image from RGB to CIELAB color space for perceptual uniformity.
- Standardizes pixel values before clustering.
- Implements K-means clustering with K-means++ initialization.
- Supports clustering with configurable values of K.
- Identifies the smallest cluster (usually representing numbers or distinct objects).
- Highlights segmented regions with clear green color.
- Includes an elbow method to estimate the optimal number of clusters K.
- Visualizes inertia plots (elbow curve) and segmented images.

---

## Getting Started

### Requirements

- Python 3.7 or higher
- The following Python packages (can be installed using pip):

```bash
numpy
pillow
matplotlib
opencv-python
scikit-image
scikit-learn
