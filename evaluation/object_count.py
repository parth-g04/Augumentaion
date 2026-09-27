"""
object_count.py
---------------
Metric 2: standalone object-count drift.

bbox_iou.compute_bbox_iou_and_drift already returns count_drift as a
byproduct of matching, so this module exists for the case where you
only have raw counts available (e.g. a lightweight counter run upstream
of a full detector) and want the drift number without doing IoU matching.
"""

from typing import List
from .interfaces import BBox2D


def compute_count_drift(orig_boxes: List[BBox2D], aug_boxes: List[BBox2D]) -> int:
    return abs(len(orig_boxes) - len(aug_boxes))