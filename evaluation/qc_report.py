"""
Layer 4 Quality Control Report
"""

import json


def build_report(
    metrics,
    thresholds,
    metadata=None
):

    if metadata is None:
        metadata = {}

    # ========================================================
    # METRICS
    # ========================================================

    bbox_iou = getattr(
        metrics,
        "mean_bbox_iou",
        None
    )

    count_drift = getattr(
        metrics,
        "object_count_drift",
        None
    )

    edge_similarity = getattr(
        metrics,
        "edge_similarity",
        None
    )

    edge_recall = getattr(
        metrics,
        "edge_preservation_recall",
        None
    )

    semantic_iou = getattr(
        metrics,
        "semantic_mask_iou",
        None
    )

    # ========================================================
    # THRESHOLDS
    # ========================================================

    bbox_threshold = (
        thresholds.min_bbox_iou
    )

    count_threshold = (
        thresholds.max_count_drift
    )

    edge_threshold = (
        thresholds.min_edge_similarity
    )

    semantic_threshold = (
        thresholds.min_semantic_mask_iou
    )

    per_object_threshold = (
        getattr(
            thresholds,
            "min_per_object_iou",
            0.60
        )
    )

    # ========================================================
    # BBOX
    # ========================================================

    bbox_available = metadata.get(
        "bbox_annotations_available",
        False
    )

    per_object = metadata.get(
        "hard_per_object_bbox"
    )

    if not bbox_available:

        bbox_check = None

        bbox_reason = (
            "Not evaluated — "
            "bounding-box annotations "
            "were not provided."
        )

    elif per_object is not None:

        bbox_check = (
            per_object["hard_bbox_pass"]
        )

        if bbox_check:

            bbox_reason = (
                "All matched objects satisfy "
                f"IoU >= {per_object_threshold:.2f} "
                "and no new objects were detected."
            )

        else:

            bbox_reason = (
                "One or more objects failed the "
                f"per-object IoU threshold of "
                f"{per_object_threshold:.2f}, "
                "or new objects were detected."
            )

    else:

        bbox_check = (
            bbox_iou >= bbox_threshold
            if bbox_iou is not None
            else None
        )

        bbox_reason = (
            f"BBox IoU {bbox_iou:.3f} >= "
            f"{bbox_threshold:.3f}"
            if bbox_check
            else
            f"BBox IoU {bbox_iou:.3f} < "
            f"{bbox_threshold:.3f}"
        )

    # ========================================================
    # OBJECT COUNT
    # ========================================================

    if count_drift is None:

        count_check = None

        count_reason = (
            "Not evaluated — object "
            "annotations were not provided."
        )

    else:

        count_check = (
            count_drift <= count_threshold
        )

        count_reason = (
            f"Count drift {count_drift} <= "
            f"{count_threshold}"
            if count_check
            else
            f"Count drift {count_drift} > "
            f"{count_threshold}"
        )

    # ========================================================
    # EDGE
    # ========================================================

    if edge_similarity is None:

        edge_check = None

        edge_reason = (
            "Not evaluated."
        )

    else:

        edge_check = (
            edge_similarity >= edge_threshold
        )

        edge_reason = (
            f"Edge similarity {edge_similarity:.3f} >= "
            f"{edge_threshold:.3f}"
            if edge_check
            else
            f"Edge similarity {edge_similarity:.3f} < "
            f"{edge_threshold:.3f}"
        )

    # ========================================================
    # SEMANTIC
    # ========================================================

    if semantic_iou is None:

        semantic_check = None

        semantic_reason = (
            "Not evaluated — segmentation "
            "masks were not provided."
        )

    else:

        semantic_check = (
            semantic_iou >= semantic_threshold
        )

        semantic_reason = (
            f"Semantic IoU {semantic_iou:.3f} >= "
            f"{semantic_threshold:.3f}"
            if semantic_check
            else
            f"Semantic IoU {semantic_iou:.3f} < "
            f"{semantic_threshold:.3f}"
        )

    # ========================================================
    # FAILURE REASONS
    # ========================================================

    failure_reasons = []

    if edge_check is False:

        failure_reasons.append(
            edge_reason
        )

    if bbox_check is False:

        failure_reasons.append(
            bbox_reason
        )

    if count_check is False:

        failure_reasons.append(
            count_reason
        )

    if semantic_check is False:

        failure_reasons.append(
            semantic_reason
        )

    # ========================================================
    # STATUS
    # ========================================================

    evaluated = [
        edge_check,
        bbox_check,
        count_check,
        semantic_check
    ]

    evaluated_only = [
        x for x in evaluated
        if x is not None
    ]

    if any(
        x is False
        for x in evaluated_only
    ):

        status = "REJECT"

    elif any(
        x is None
        for x in evaluated
    ):

        status = "WARNING"

    else:

        status = "PASS"

    # ========================================================
    # FINAL REPORT
    # ========================================================

    return {

        "status": status,

        "passed_checks": {

            "structural_edges":
                edge_check,

            "hard_per_object_bbox":
                bbox_check,

            "object_count":
                count_check,

            "semantic_consistency":
                semantic_check
        },

        "metrics": {

            "edge_similarity":
                edge_similarity,

            "edge_preservation_recall":
                edge_recall,

            "mean_bbox_iou":
                bbox_iou,

            "object_count_drift":
                count_drift,

            "semantic_mask_iou":
                semantic_iou,

            # FID is calculated separately at
            # dataset level.
            "fid":
                metadata.get("fid")
        },

        "thresholds": {

            "edge_similarity":
                edge_threshold,

            "bbox_iou":
                bbox_threshold,

            "per_object_iou":
                per_object_threshold,

            "object_count_drift":
                count_threshold,

            "semantic_iou":
                semantic_threshold,

            "fid":
                15.0
        },

        "reasons": {

            "structural_edges":
                edge_reason,

            "bbox":
                bbox_reason,

            "object_count":
                count_reason,

            "semantic_iou":
                semantic_reason,

            "fid":
                metadata.get(
                    "fid_reason",
                    "FID not evaluated."
                )
        },

        "failure_reasons":
            failure_reasons,

        "metadata":
            metadata
    }