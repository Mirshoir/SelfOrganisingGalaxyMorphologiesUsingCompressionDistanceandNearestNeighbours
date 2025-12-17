# src/knn_embed.py

import numpy as np
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram


def run_pca_tsne(X: np.ndarray, y: np.ndarray):
    # ---------------------------------------------------
    # PCA
    # ---------------------------------------------------
    print("\nRunning PCA on full feature matrix...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="coolwarm", s=10)
    plt.title("PCA Projection of NCD Feature Space")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(scatter, label="Class")
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------
    # t-SNE
    # ---------------------------------------------------
    print("\nRunning t-SNE (this may take a bit)...")
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, init="random")
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="coolwarm", s=10)
    plt.title("t-SNE Projection of NCD Feature Space")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.colorbar(scatter, label="Class")
    plt.tight_layout()
    plt.show()


def run_dendrogram(X: np.ndarray, subset: int = 200):
    print(f"\nRunning hierarchical clustering on subset (first {subset} images)...")
    X_sub = X[:subset]

    Z = linkage(X_sub, method="ward")

    plt.figure(figsize=(12, 5))
    dendrogram(Z, truncate_mode="level", p=5)
    plt.title("Hierarchical Clustering (Dendrogram) — First 200 Images")
    plt.xlabel("Sample index (subset)")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    X_path = root / "data" / "processed" / "X_ncd.npy"
    y_path = root / "data" / "processed" / "y_labels.npy"

    X = np.load(X_path)
    y = np.load(y_path)

    run_pca_tsne(X, y)
    run_dendrogram(X, subset=200)
