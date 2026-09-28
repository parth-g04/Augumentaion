"""
edge_similarity.py
------------------
Metric 3: Sobel-gradient structural edge metrics.

Extracted verbatim from QualityGate.compute_edge_metrics. Uses Sobel
gradients (not Canny) thresholded at a fixed normalized magnitude, then
compares edge maps two ways:
  - edge_iou: symmetric overlap (penalizes both missing AND spurious edges)
  - edge_preservation_recall: one-directional — what fraction of ORIGINAL
    edge pixels survived, tolerant of new weather-texture edges being added
"""

from typing import Tuple
import numpy as np
from scipy import ndimage


def get_edges(img: np.ndarray, threshold: float = 0.15) -> np.ndarray:
    gray = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    gx = ndimage.sobel(gray, axis=1)
    gy = ndimage.sobel(gray, axis=0)
    grad = np.hypot(gx, gy)
    max_g = np.max(grad)
    if max_g > 1e-6:
        grad /= max_g
    return (grad > threshold).astype(bool)


def compute_edge_metrics(orig_rgb: np.ndarray, aug_rgb: np.ndarray) -> Tuple[float, float]:
    """Returns (edge_iou, edge_preservation_recall)."""
    e_orig = get_edges(orig_rgb)
    e_aug = get_edges(aug_rgb)

    orig_count = e_orig.sum()
    inter = np.logical_and(e_orig, e_aug).sum()
    union = np.logical_or(e_orig, e_aug).sum()

    edge_iou = float(inter / union) if union > 0 else 1.0
    edge_recall = float(inter / orig_count) if orig_count > 0 else 1.0

    return edge_iou, edge_recall