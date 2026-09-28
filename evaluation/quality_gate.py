from dataclasses import dataclass

from .bbox_iou import (
    compute_bbox_iou_and_drift,
    compute_per_object_bbox_check
)

from .edge_similarity import (
    compute_edge_metrics
)

from .semantic_iou import (
    compute_mask_iou
)

from .interfaces import MetricScores

from .qc_report import build_report


@dataclass
class QualityGateThresholds:

    min_bbox_iou: float = 0.60

    max_count_drift: int = 2

    min_edge_similarity: float = 0.15

    min_semantic_mask_iou: float = 0.50

    # Hard per-object IoU
    min_per_object_iou: float = 0.60

    # Maximum allowed object movement in pixels
    max_object_displacement: float = 50.0


class QualityGate:

    def __init__(
        self,
        thresholds=None
    ):

        self.thresholds = (
            thresholds
            or QualityGateThresholds()
        )

    def evaluate(
        self,
        orig_rgb,
        aug_rgb,
        orig_boxes,
        aug_boxes,
        orig_mask=None,
        aug_mask=None,
        metadata=None
    ):

        # ====================================================
        # BBOX
        # ====================================================

        bbox_iou, count_drift = (
            compute_bbox_iou_and_drift(
                orig_boxes,
                aug_boxes
            )
        )

        per_object = None

        if orig_boxes and aug_boxes:

            per_object = (
                compute_per_object_bbox_check(
                    orig_boxes,
                    aug_boxes,
                    min_iou=(
                        self.thresholds
                        .min_per_object_iou
                    )
                )
            )

        # ====================================================
        # EDGE
        # ====================================================

        edge_iou, edge_recall = (
            compute_edge_metrics(
                orig_rgb,
                aug_rgb
            )
        )

        # ====================================================
        # SEMANTIC
        # ====================================================

        mask_iou = None

        if (
            orig_mask is not None
            and aug_mask is not None
        ):

            mask_iou = compute_mask_iou(
                orig_mask,
                aug_mask
            )

        # ====================================================
        # METRICS
        # ====================================================

        metrics = MetricScores(

            mean_bbox_iou=(
                round(
                    bbox_iou,
                    3
                )
                if orig_boxes and aug_boxes
                else None
            ),

            object_count_drift=(
                count_drift
                if orig_boxes and aug_boxes
                else None
            ),

            edge_similarity=round(
                edge_iou,
                3
            ),

            edge_preservation_recall=round(
                edge_recall,
                3
            ),

            semantic_mask_iou=(
                round(
                    mask_iou,
                    3
                )
                if mask_iou is not None
                else None
            )
        )

        # ====================================================
        # METADATA
        # ====================================================

        if metadata is None:
            metadata = {}

        metadata["bbox_annotations_available"] = (
            bool(orig_boxes and aug_boxes)
        )

        metadata["hard_per_object_bbox"] = (
            per_object
        )

        # ====================================================
        # REPORT
        # ====================================================

        return build_report(
            metrics=metrics,
            thresholds=self.thresholds,
            metadata=metadata
        )