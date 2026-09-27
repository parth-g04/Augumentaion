"""
depth_consistency.py
---------------------
Metric 5 (optional): depth_correlation.

NOT currently called from QualityGate.evaluate() — MetricScores.depth_correlation
stays None until Person 4's manifest supplies per-sample depth maps and this
is wired into evaluate()'s signature (add orig_depth/aug_depth params,
mirroring how orig_mask/aug_mask are already handled).

Uses Spearman rank correlation rather than absolute difference because
depth re-estimated on a re-lit (fog/night) image won't share the original's
absolute scale — only relative near/far ordering is meaningful to compare.
"""

import numpy as np
from scipy.stats import spearmanr


def compute_depth_correlation(orig_depth: np.ndarray, aug_depth: np.ndarray) -> float:
    if orig_depth.shape != aug_depth.shape:
        raise ValueError(
            f"Depth map shape mismatch: {orig_depth.shape} vs {aug_depth.shape}"
        )
    a = orig_depth.flatten()
    b = aug_depth.flatten()
    corr, _ = spearmanr(a, b)
    return float(corr) if corr is not None else 0.0