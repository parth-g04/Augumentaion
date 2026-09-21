"""Quality Gate and Consistency Checker for Layer 4.
Evaluates bounding-box consistency, object count drift, and structural edge similarity
to produce an empirical PASS/REJECT decision on augmented frames.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import ndimage

from .interfaces import BBox2D, MetricScores, QCReport, QCStatus


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

    def compute_edge_similarity(self, orig_rgb: np.ndarray, aug_rgb: np.ndarray) -> float:
        """Computes structural edge intersection over union between two frames."""
        def get_edges(img: np.ndarray) -> np.ndarray:
            gray = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
            gx = ndimage.sobel(gray, axis=1)
            gy = ndimage.sobel(gray, axis=0)
            grad = np.hypot(gx, gy)
            max_g = np.max(grad)
            if max_g > 1e-6:
                grad /= max_g
            return (grad > 0.15).astype(bool)

        e_orig = get_edges(orig_rgb)
        e_aug = get_edges(aug_rgb)

        inter = np.logical_and(e_orig, e_aug).sum()
        union = np.logical_or(e_orig, e_aug).sum()

        if union == 0:
            return 1.0
        return float(inter / union)

    def match_and_score_bboxes(
        self,
        orig_boxes: List[BBox2D],
        aug_boxes: List[BBox2D]
    ) -> Tuple[float, int]:
        """Greedy matching of bounding boxes by IoU.
        Returns:
            (mean_iou, count_drift)
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

            if best_idx >= 0 and best_iou > 0.1:
                ious.append(best_iou)
                unmatched_aug.pop(best_idx)
            else:
                ious.append(0.0)

        mean_iou = float(np.mean(ious)) if ious else 0.0
        return mean_iou, count_drift

    def evaluate(
        self,
        orig_rgb: np.ndarray,
        aug_rgb: np.ndarray,
        orig_boxes: List[BBox2D],
        aug_boxes: List[BBox2D],
        orig_mask: Optional[np.ndarray] = None,
        aug_mask: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, any]] = None
    ) -> QCReport:
        """Performs full quality gate check and issues a PASS / REJECT verdict."""
        mean_iou, count_drift = self.match_and_score_bboxes(orig_boxes, aug_boxes)
        edge_sim = self.compute_edge_similarity(orig_rgb, aug_rgb)

        mask_iou = None
        if orig_mask is not None and aug_mask is not None:
            inter = np.logical_and(orig_mask > 0, aug_mask > 0).sum()
            union = np.logical_or(orig_mask > 0, aug_mask > 0).sum()
            mask_iou = float(inter / union) if union > 0 else 1.0

        metrics = MetricScores(
            mean_bbox_iou=round(mean_iou, 3),
            object_count_drift=count_drift,
            edge_similarity=round(edge_sim, 3),
            semantic_mask_iou=round(mask_iou, 3) if mask_iou is not None else None
        )

        passed_checks = {
            "bbox_consistency": mean_iou >= self.thresholds.min_bbox_iou,
            "count_drift_within_tolerance": count_drift <= self.thresholds.max_count_drift,
            "structural_edge_preserved": edge_sim >= self.thresholds.min_edge_similarity
        }
        if mask_iou is not None:
            passed_checks["mask_iou_consistent"] = mask_iou >= self.thresholds.min_semantic_mask_iou

        failure_reasons = []
        if not passed_checks["bbox_consistency"]:
            failure_reasons.append(
                f"BBox consistency IoU {metrics.mean_bbox_iou:.3f} below threshold {self.thresholds.min_bbox_iou}"
            )
        if not passed_checks["count_drift_within_tolerance"]:
            failure_reasons.append(
                f"Object count drift {metrics.object_count_drift} exceeds max tolerance {self.thresholds.max_count_drift}"
            )
        if not passed_checks["structural_edge_preserved"]:
            failure_reasons.append(
                f"Edge similarity {metrics.edge_similarity:.3f} below threshold {self.thresholds.min_edge_similarity}"
            )
        if mask_iou is not None and not passed_checks["mask_iou_consistent"]:
            failure_reasons.append(
                f"Semantic mask IoU {mask_iou:.3f} below threshold {self.thresholds.min_semantic_mask_iou}"
            )

        status: QCStatus = "PASS" if all(passed_checks.values()) else "REJECT"

        return QCReport(
            status=status,
            metrics=metrics,
            passed_checks=passed_checks,
            failure_reasons=failure_reasons,
            metadata=metadata or {}
        )
