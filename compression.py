# compression.py
# ============================================================
# Image → PNG bytes + Zstandard-based NCD
# ============================================================

from typing import List, Tuple, Iterable
import io
import numpy as np
from PIL import Image, ImageFilter
import zstandard as zstd


# ============================================================
# IMAGE → PNG BYTES (CONFIGURABLE PREPROCESSING)
# ============================================================
def image_to_bytes_array(
    img_array: np.ndarray,
    size: Tuple[int, int] = (64, 64),
    mode: str = "L",
    threshold_percent: float | None = None,
    blur_radius: float | None = None,
) -> bytes:
    """
    Convert a NumPy image to PNG bytes with optional preprocessing.

    Steps (if enabled):
    1. Normalize grayscale image from 0–255 → 0–1
    2. Threshold: set bottom X% pixels to zero
    3. Gaussian smoothing
    4. Resize
    5. Encode to PNG bytes

    Args:
        img_array: (H, W) or (H, W, C)
        size: target resize (width, height)
        mode: "L" or "RGB"
        threshold_percent: e.g. 90.0 (None disables thresholding)
        blur_radius: Gaussian blur radius (None disables smoothing)

    Returns:
        PNG-encoded bytes
    """

    # ---------------------------------------------------------
    # 1. Convert to float & normalize
    # ---------------------------------------------------------
    img = img_array.astype(np.float32) / 255.0

    # ---------------------------------------------------------
    # 2. Brightness-rank threshold (optional)
    # ---------------------------------------------------------
    if threshold_percent is not None:
        thr = np.percentile(img, threshold_percent)
        img = np.where(img >= thr, img, 0.0)

    # ---------------------------------------------------------
    # 3. Back to uint8 for PIL
    # ---------------------------------------------------------
    img_uint8 = (img * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8).convert(mode)

    # ---------------------------------------------------------
    # 4. Gaussian smoothing (optional)
    # ---------------------------------------------------------
    if blur_radius is not None and blur_radius > 0:
        pil_img = pil_img.filter(
            ImageFilter.GaussianBlur(radius=blur_radius)
        )

    # ---------------------------------------------------------
    # 5. Resize
    # ---------------------------------------------------------
    pil_img = pil_img.resize(size)

    # ---------------------------------------------------------
    # 6. Encode to PNG bytes
    # ---------------------------------------------------------
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


# ============================================================
# MULTI-SCALE SUPPORT (3 < N < 10)
# ============================================================
def image_to_bytes_multiscale(
    img_array: np.ndarray,
    size: Tuple[int, int],
    mode: str,
    threshold_percent: float | None,
    blur_scales: Iterable[float],
) -> List[bytes]:
    """
    Generate PNG bytes for multiple smoothing scales.

    Used for multi-scale aggregation experiments.
    """
    out = []
    for r in blur_scales:
        out.append(
            image_to_bytes_array(
                img_array,
                size=size,
                mode=mode,
                threshold_percent=threshold_percent,
                blur_radius=r,
            )
        )
    return out


def convert_images_to_bytes(
    images: np.ndarray,
    size: Tuple[int, int] = (64, 64),
    mode: str = "L",
    threshold_percent: float | None = None,
    blur_radius: float | None = None,
) -> List[bytes]:
    """
    Convert a batch of images to PNG bytes (single-scale).
    """
    return [
        image_to_bytes_array(
            img,
            size=size,
            mode=mode,
            threshold_percent=threshold_percent,
            blur_radius=blur_radius,
        )
        for img in images
    ]


# ============================================================
# ZSTANDARD COMPRESSION (MULTI-THREADED)
# ============================================================
_ZSTD_COMPRESSOR = zstd.ZstdCompressor(
    level=3,
    threads=0,  # use all available cores
)


def zstd_size(data: bytes) -> int:
    return len(_ZSTD_COMPRESSOR.compress(data))


# ============================================================
# NORMALIZED COMPRESSION DISTANCE (NCD)
# ============================================================
def ncd_bytes(bx: bytes, by: bytes) -> float:
    """
    NCD(x, y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))
    """
    Cx = zstd_size(bx)
    Cy = zstd_size(by)
    Cxy = zstd_size(bx + by)
    return (Cxy - min(Cx, Cy)) / max(Cx, Cy)
