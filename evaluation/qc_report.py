"""
qc_report.py
------------
Builds passed_checks / failure_reasons / QCReport from already-computed
metrics and thresholds. Extracted verbatim from the second half of
QualityGate.evaluate() so the pass/fail RULE logic is decoupled from
metric COMPUTATION and independently testable (e.g. you can unit-test
threshold edge cases with hand-crafted MetricScores, no images needed).
"""

from typing import Any, Dict, List, Optional

from .interfaces import MetricScores, QCReport, QCStatus


def build_report(
    metrics: MetricScores,
    thresholds,  # QualityGateThresholds, defined in quality_gate.py
    metadata: Optional[Dict[str, Any]] = None
) -> QCReport:
    passed_checks: Dict[str, bool] = {
        "bbox_consistency": metrics.mean_bbox_iou >= thresholds.min_bbox_iou,
        "count_drift_within_tolerance": (
            metrics.object_count_drift <= thresholds.max_count_drift
        ),
        "structural_edge_preserved": (
            metrics.edge_preservation_recall >= thresholds.min_edge_similarity
        ),
    }
    if metrics.semantic_mask_iou is not None:
        passed_checks["mask_iou_consistent"] = (
            metrics.semantic_mask_iou >= thresholds.min_semantic_mask_iou
        )

    failure_reasons: List[str] = []
    if not passed_checks["bbox_consistency"]:
        failure_reasons.append(
            f"BBox consistency IoU {metrics.mean_bbox_iou:.3f} below threshold "
            f"{thresholds.min_bbox_iou}"
        )
    if not passed_checks["count_drift_within_tolerance"]:
        failure_reasons.append(
            f"Object count drift {metrics.object_count_drift} exceeds max "
            f"tolerance {thresholds.max_count_drift}"
        )
    if not passed_checks["structural_edge_preserved"]:
        failure_reasons.append(
            f"Edge similarity {metrics.edge_similarity:.3f} below threshold "
            f"{thresholds.min_edge_similarity}"
        )
    if (metrics.semantic_mask_iou is not None
            and not passed_checks["mask_iou_consistent"]):
        failure_reasons.append(
            f"Semantic mask IoU {metrics.semantic_mask_iou:.3f} below threshold "
            f"{thresholds.min_semantic_mask_iou}"
        )

    status: QCStatus = "PASS" if all(passed_checks.values()) else "REJECT"

    return QCReport(
        status=status,
        metrics=metrics,
        passed_checks=passed_checks,
        failure_reasons=failure_reasons,
        metadata=metadata or {}
    )