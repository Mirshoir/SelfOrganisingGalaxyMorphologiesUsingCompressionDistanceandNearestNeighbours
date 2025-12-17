🌌 Compression-Based Galaxy Morphology

A feature-free approach to galaxy morphology classification using compression distance, anchor representations, and nearest-neighbour learning.

This project explores how galaxy images can be compared and classified without hand-crafted features or deep neural networks, using ideas from information theory.

1. Introduction

Classifying galaxy shapes is an important task in astronomy.
Galaxies have different forms such as spiral, elliptical, and irregular.
Traditionally, this problem is solved using hand-crafted image features or deep learning models.

In this project, we study a different idea.
Instead of extracting visual features, we measure how well images compress together.
The main assumption is simple:

Images with similar galaxy shapes compress better together than different ones.

This approach does not need feature engineering or large neural networks.

2. Dataset

We use a dataset of grayscale galaxy images.
Each image has a known morphology label.

Images are stored in folders, where each folder represents one galaxy class.
To avoid class imbalance, we optionally limit the number of images per class.

3. Main Idea: Compression Distance

Compression algorithms remove repeated patterns.
If two images are similar, compressing them together saves more space.

We use Normalised Compression Distance (NCD) to measure similarity between images.

In simple terms:

Small NCD → images are similar

Large NCD → images are different

This gives us a way to compare images without extracting visual features.


4. Anchor-Based Representation

Comparing every image with every other image is slow.
To solve this, we use anchor images.

Steps:

Select a small number of anchor images per class

Measure the compression distance from each image to all anchors

Use these distances as a feature vector

This creates a fixed-length representation for every image.


5. Classification Methods

We test two classifiers:

5.1 k-Nearest Neighbours (kNN)

Classifies an image using its closest neighbours

Works well with distance-based features

Simple and interpretable

5.2 Support Vector Machine (SVM)

Uses a nonlinear decision boundary

Can separate complex patterns

Sometimes sensitive to preprocessing

Both models are trained and tested using the same data splits

6. Image Preprocessing Experiments

We compare two cases:

6.1 Before Preprocessing (Baseline)

Raw grayscale images

Only resizing and compression

6.2 After Preprocessing

We apply three steps:

Normalize pixel values from 0–255 to 0–1

Keep only the brightest 10% of pixels

Apply Gaussian smoothing (radius 3–9 pixels)

This removes background noise and highlights galaxy structure.


7. Experiments
7.1 Two-Stage Comparison

Before smoothing

After smoothing

Both stages use the same anchors and classifiers.

7.2 Resolution Sensitivity

We test different image sizes:

32 × 32

64 × 64

128 × 128

This checks if the method depends on image resolution.

7.3 Automatic Ablation Study

We run many experiments automatically by changing:

Image resolution

Threshold percentage

Smoothing radius

All results are saved to a CSV file for analysis.

8. Results
8.1 Classification Accuracy

kNN and SVM give similar accuracy

kNN is more stable across settings

SVM sometimes overfits after heavy preprocessing

Surprisingly, preprocessing does not always improve accuracy.

8.2 Visualization Results

We use:

PCA

t-SNE

Dendrograms

Nearest-neighbour image galleries

These show that galaxies with similar shapes are grouped together.

8.3 Resolution Analysis

Accuracy stays similar across different resolutions.
This means compression distance captures important structure even at low resolution.

9. Discussion

Key observations:

Compression already removes noise naturally

Extra smoothing does not always help

The representation method matters more than the compressor

Anchors make the method practical and scalable

This explains why results before and after smoothing can be very similar.

10. Limitations and Future Work

Limitations:

Compression is slower than simple features

Anchor selection can affect results

Future improvements:

Better anchor selection strategies

Combining compression with learned features

Using multi-band (color) galaxy images


11. Conclusion

This project shows that galaxy morphology can be classified using compression-based similarity without traditional features or deep learning.

The method is:

Simple

Interpretable

Robust across resolutions

Compression distance is a strong alternative for exploratory and low-data scientific tasks.
