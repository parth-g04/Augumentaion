"""Quality Gate and Consistency Checker for Layer 4.
Evaluates bounding-box consistency, object count drift, and structural edge similarity
to produce an empirical PASS/REJECT decision on augmented frames.

Refactored to delegate metric computation to dedicated modules
(bbox_iou.py, edge_similarity.py, semantic_iou.py, qc_report.py) instead
of computing everything inline. QualityGate.evaluate()'s signature and
behavior are UNCHANGED — anything already calling it (e.g. an
orchestration script) does not need to change.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

from .interfaces import BBox2D, MetricScores, QCReport
from .bbox_iou import compute_bbox_iou_and_drift
from .edge_similarity import compute_edge_metrics
from .semantic_iou import compute_mask_iou
from .qc_report import build_report


@dataclass
class QualityGateThresholds:
    """Configurable empirical thresholds for Quality Gate acceptance."""
    min_bbox_iou: float = 0.60
    max_count_drift: int = 2     # Tolerates small dropouts under extreme rain/night
    min_edge_similarity: float = 0.35
    min_semantic_mask_iou: float = 0.50


class QualityGate:
    """Evaluates consistency between original reference render and augmented variant."""

    def __init__(self, thresholds: Optional[QualityGateThresholds] = None):
        self.thresholds = thresholds or QualityGateThresholds()

    def evaluate(
        self,
        orig_rgb: np.ndarray,
        aug_rgb: np.ndarray,
        orig_boxes: List[BBox2D],
        aug_boxes: List[BBox2D],
        orig_mask: Optional[np.ndarray] = None,
        aug_mask: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> QCReport:
        """Performs full quality gate check and issues a PASS / REJECT verdict."""
        mean_iou, count_drift = compute_bbox_iou_and_drift(orig_boxes, aug_boxes)
        edge_iou, edge_recall = compute_edge_metrics(orig_rgb, aug_rgb)

        mask_iou = None
        if orig_mask is not None and aug_mask is not None:
            mask_iou = compute_mask_iou(orig_mask, aug_mask)

        metrics = MetricScores(
            mean_bbox_iou=round(mean_iou, 3),
            object_count_drift=count_drift,
            edge_similarity=round(edge_iou, 3),
            edge_preservation_recall=round(edge_recall, 3),
            semantic_mask_iou=round(mask_iou, 3) if mask_iou is not None else None,
        )

        return build_report(metrics=metrics, thresholds=self.thresholds, metadata=metadata)