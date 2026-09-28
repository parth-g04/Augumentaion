from typing import Dict

import numpy as np
from scipy import ndimage


def _luminance(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float64)

    return (
        0.299 * img[..., 0]
        + 0.587 * img[..., 1]
        + 0.114 * img[..., 2]
    )


def _log_luminance_edges(
    img: np.ndarray,
    keep: float = 0.08,
) -> np.ndarray:

    luminance = _luminance(img)

    luminance = ndimage.gaussian_filter(
        luminance,
        sigma=1.0
    )

    log_luminance = np.log1p(luminance)

    gx = ndimage.sobel(
        log_luminance,
        axis=1
    )

    gy = ndimage.sobel(
        log_luminance,
        axis=0
    )

    gradient = np.hypot(gx, gy)

    threshold = np.quantile(
        gradient,
        1.0 - keep
    )

    return gradient >= threshold


def compute_edge_metrics_robust(
    orig_rgb: np.ndarray,
    aug_rgb: np.ndarray,
    keep: float = 0.08,
    tolerance: int = 3,
    lit_threshold: float = 8.0,
) -> Dict[str, float]:

    if orig_rgb.shape[:2] != aug_rgb.shape[:2]:
        raise ValueError(
            f"Image size mismatch: "
            f"{orig_rgb.shape[:2]} vs "
            f"{aug_rgb.shape[:2]}"
        )

    original_edges = _log_luminance_edges(
        orig_rgb,
        keep
    )

    generated_edges = _log_luminance_edges(
        aug_rgb,
        keep
    )

    # Allow small spatial shifts.
    generated_edges_tolerant = ndimage.maximum_filter(
        generated_edges,
        size=2 * tolerance + 1
    )

    original_count = int(original_edges.sum())

    if original_count == 0:
        recall = 1.0
    else:
        overlap = np.logical_and(
            original_edges,
            generated_edges_tolerant
        ).sum()

        recall = float(overlap / original_count)

    chance_level = float(
        generated_edges_tolerant.mean()
    )

    if chance_level < 1.0:
        corrected = (
            recall - chance_level
        ) / (
            1.0 - chance_level
        )
    else:
        corrected = 0.0

    corrected = float(
        np.clip(corrected, 0.0, 1.0)
    )

    smooth_luminance = ndimage.gaussian_filter(
        _luminance(aug_rgb),
        sigma=3.0
    )

    lit_fraction = float(
        (
            smooth_luminance >= lit_threshold
        ).mean()
    )

    return {
        "edge_recall_tolerant": round(
            recall,
            4
        ),
        "chance_level": round(
            chance_level,
            4
        ),
        "edge_recall_chance_corrected": round(
            corrected,
            4
        ),
        "lit_fraction": round(
            lit_fraction,
            4
        ),
    }