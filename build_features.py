# build_features.py

from __future__ import annotations
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import numpy as np

from compression import ncd_bytes


# ------------------------------------------------------------
# Anchor selection
# ------------------------------------------------------------
def select_anchors_auto(
    labels: np.ndarray,
    num_anchors_per_class: int,
    seed: int = 42,
) -> List[int]:
    """
    AUTO mode:
    Randomly select anchors from dataset, per class.
    """
    rng = np.random.default_rng(seed)
    anchors: List[int] = []

    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue

        rng.shuffle(idx)
        take = min(num_anchors_per_class, len(idx))
        anchors.extend(idx[:take].tolist())

    return anchors


def load_manual_anchor_bytes(
    anchor_root: str,
    class_names: List[str],
    image_to_bytes_fn,
) -> Tuple[List[bytes], List[str]]:
    """
    MANUAL mode:
    Load anchor images from anchor_root/class_name/*.jpg

    Returns:
        anchor_bytes: list of PNG bytes
        anchor_labels: list of class names (same length)
    """
    root = Path(anchor_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Anchor root not found: {root}")

    anchor_bytes: List[bytes] = []
    anchor_labels: List[str] = []

    for cname in class_names:
        class_dir = root / cname
        if not class_dir.exists():
            continue

        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue

            b = image_to_bytes_fn(img_path)
            anchor_bytes.append(b)
            anchor_labels.append(cname)

    if not anchor_bytes:
        raise ValueError("No manual anchors found.")

    return anchor_bytes, anchor_labels


# ------------------------------------------------------------
# Feature matrix
# ------------------------------------------------------------
def build_feature_matrix(
    images_bytes: List[bytes],
    anchor_bytes: List[bytes],
) -> np.ndarray:
    """
    Compute NCD features: (N samples × M anchors)
    """
    N = len(images_bytes)
    M = len(anchor_bytes)

    X = np.zeros((N, M), dtype=np.float32)

    for j, a in enumerate(anchor_bytes):
        for i in range(N):
            X[i, j] = ncd_bytes(images_bytes[i], a)

    return X


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------
def compute_features_with_anchors(
    images_bytes: List[bytes],
    labels: np.ndarray,
    class_names: List[str],
    anchor_mode: str,
    anchors_per_class: int,
    anchor_root: Optional[str] = None,
    seed: int = 42,
    image_to_bytes_fn=None,
) -> Tuple[np.ndarray, Dict]:
    """
    Unified anchor feature builder.

    Returns:
        X: feature matrix
        info: metadata about anchors
    """

    if anchor_mode == "auto":
        anchor_indices = select_anchors_auto(
            labels, anchors_per_class, seed
        )
        anchor_bytes = [images_bytes[i] for i in anchor_indices]
        info = {
            "mode": "auto",
            "num_anchors": len(anchor_bytes),
            "anchors_per_class": anchors_per_class,
        }

    elif anchor_mode == "manual":
        if anchor_root is None or image_to_bytes_fn is None:
            raise ValueError("manual mode requires anchor_root and image_to_bytes_fn")

        anchor_bytes, anchor_labels = load_manual_anchor_bytes(
            anchor_root, class_names, image_to_bytes_fn
        )
        info = {
            "mode": "manual",
            "num_anchors": len(anchor_bytes),
            "anchor_root": anchor_root,
        }

    else:
        raise ValueError(f"Unknown anchor_mode: {anchor_mode}")

    X = build_feature_matrix(images_bytes, anchor_bytes)
    return X, info
