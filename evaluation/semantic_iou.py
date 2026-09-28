"""
semantic_iou.py
---------------
Metric 4: binary semantic mask IoU.

Extracted verbatim from QualityGate.evaluate's inline mask comparison.
Treats any nonzero pixel as "foreground" — for multi-class per-class mIoU
instead of a single binary overlap, extend this with a per-class loop
(see the class-aware version noted in the docstring below) once semantic
masks with multiple classes are wired in from Person 4's manifest.
"""

import numpy as np


def compute_mask_iou(orig_mask: np.ndarray, aug_mask: np.ndarray) -> float:
    """Binary foreground IoU (matches current QualityGate.evaluate behavior)."""
    inter = np.logical_and(orig_mask > 0, aug_mask > 0).sum()
    union = np.logical_or(orig_mask > 0, aug_mask > 0).sum()
    return float(inter / union) if union > 0 else 1.0


def compute_per_class_mask_iou(orig_mask: np.ndarray, aug_mask: np.ndarray) -> float:
    """
    Optional upgrade: per-class IoU averaged (true mIoU), for when masks
    carry class ids rather than a single foreground/background split.
    Not currently called by QualityGate.evaluate — swap it in if/when
    class-indexed semantic masks become available.
    """
    classes = np.unique(orig_mask)
    ious = []
    for c in classes:
        if c == 0:
            continue
        gt_c = orig_mask == c
        pred_c = aug_mask == c
        inter = np.logical_and(gt_c, pred_c).sum()
        union = np.logical_or(gt_c, pred_c).sum()
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 1.0