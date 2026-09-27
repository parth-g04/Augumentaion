"""
bbox_iou.py
-----------
Metric 1 (+ drift): bounding-box matching and mean IoU.

Extracted verbatim from QualityGate.match_and_score_bboxes so it's
independently testable and reusable elsewhere (e.g. downstream_bench.py's
compute_ap50 does its own similar matching inline and could import this
instead in a future cleanup).
"""

from typing import List, Tuple
import numpy as np

from .interfaces import BBox2D


def compute_bbox_iou_and_drift(
    orig_boxes: List[BBox2D],
    aug_boxes: List[BBox2D],
    min_match_iou: float = 0.1
) -> Tuple[float, int]:
    """Greedy same-class matching between orig_boxes and aug_boxes.
    Returns (mean_iou, count_drift). min_match_iou=0.1 matches the
    original inline implementation's matching floor.
    """
    count_drift = abs(len(orig_boxes) - len(aug_boxes))
    if not orig_boxes and not aug_boxes:
        return 1.0, 0
    if not orig_boxes or not aug_boxes:
        return 0.0, count_drift

    ious = []
    unmatched_aug = list(aug_boxes)

    for b_orig in orig_boxes:
        best_iou = 0.0
        best_idx = -1
        for idx, b_aug in enumerate(unmatched_aug):
            if b_aug.class_name == b_orig.class_name:
                iou = b_orig.compute_iou(b_aug)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

        if best_idx >= 0 and best_iou > min_match_iou:
            ious.append(best_iou)
            unmatched_aug.pop(best_idx)
        else:
            ious.append(0.0)

    mean_iou = float(np.mean(ious)) if ious else 0.0
    return mean_iou, count_drift