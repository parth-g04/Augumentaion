from typing import List

from .interfaces import BBox2D


def compute_count_drift(
    orig_boxes: List[BBox2D],
    aug_boxes: List[BBox2D],
) -> int:

    return abs(len(orig_boxes) - len(aug_boxes))