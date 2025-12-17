# preprocess.py
# ============================================================
# Image loading + preprocessing for Galaxy Morphology project
# ============================================================

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageFilter


IMG_EXTS = (".png", ".jpg", ".jpeg")


# ============================================================
# Utility
# ============================================================
def _is_image(fname: str) -> bool:
    return fname.lower().endswith(IMG_EXTS)


# ============================================================
# MAIN LOADER (WITH LABELS)
# ============================================================
def load_images_and_labels_from_folder(
    root_dir: str,
    mode: str = "L",
    max_per_class: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Folder structure:
        root_dir/
            class1/*.jpg
            class2/*.jpg

    Returns:
        images: (N, H, W)
        labels: (N,)
        class_names: list[str]
    """

    root = Path(root_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    images: List[np.ndarray] = []
    labels: List[int] = []

    class_names = sorted(
        d.name for d in root.iterdir() if d.is_dir()
    )
    if not class_names:
        raise ValueError("No class subfolders found")

    class_to_id = {c: i for i, c in enumerate(class_names)}

    for cname in class_names:
        class_dir = root / cname
        files = sorted(
            f for f in class_dir.iterdir()
            if f.is_file() and _is_image(f.name)
        )

        if max_per_class is not None:
            files = files[:max_per_class]

        for f in files:
            img = Image.open(f).convert(mode)
            images.append(np.array(img))
            labels.append(class_to_id[cname])

    if not images:
        raise ValueError("No images loaded")

    return (
        np.stack(images),
        np.array(labels, dtype=np.int64),
        class_names,
    )


# ============================================================
# BASELINE LOADER (no labels, backward-compatible)
# ============================================================
def load_images_from_folder(
    root_dir: str,
    mode: str = "L",
) -> np.ndarray:
    images, _, _ = load_images_and_labels_from_folder(
        root_dir, mode=mode
    )
    return images


# ============================================================
# CORE PREPROCESSING (YOUR SPEC)
# ============================================================
def preprocess_single_image(
    img_array: np.ndarray,
    threshold_percent: float = 90.0,
) -> np.ndarray:
    """
    Steps:
    1. Normalize 0–255 → 0–1
    2. Rank pixels by brightness
    3. Zero bottom threshold_percent %
    """

    # Normalize
    img = img_array.astype(np.float32) / 255.0

    # Rank-based threshold
    thresh = np.percentile(img, threshold_percent)
    img = np.where(img >= thresh, img, 0.0)

    return (img * 255).astype(np.uint8)


# ============================================================
# MULTI-SCALE SMOOTHING
# ============================================================
def smooth_image_multiscale(
    img_uint8: np.ndarray,
    scales: List[int],
) -> List[np.ndarray]:
    """
    Apply Gaussian smoothing at multiple pixel scales.
    scales: e.g. [3,4,5,6,7,8,9]
    """

    pil = Image.fromarray(img_uint8).convert("L")
    outputs = []

    for s in scales:
        blurred = pil.filter(ImageFilter.GaussianBlur(radius=s))
        outputs.append(np.array(blurred))

    return outputs
