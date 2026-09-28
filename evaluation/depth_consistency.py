import numpy as np

from scipy.stats import spearmanr


def compute_depth_correlation(
    orig_depth: np.ndarray,
    aug_depth: np.ndarray,
) -> float:

    if orig_depth.shape != aug_depth.shape:
        raise ValueError(
            f"Depth map shape mismatch: "
            f"{orig_depth.shape} vs "
            f"{aug_depth.shape}"
        )

    original = orig_depth.flatten()
    generated = aug_depth.flatten()

    correlation, _ = spearmanr(
        original,
        generated
    )

    if correlation is None:
        return 0.0

    if np.isnan(correlation):
        return 0.0

    return float(correlation)