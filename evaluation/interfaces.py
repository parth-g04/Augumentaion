"""Quality Gate and Consistency evaluation schemas."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


QCStatus = Literal["PASS", "REJECT", "WARNING"]


@dataclass
class BBox2D:
    """2D bounding box with coordinate normalization and IoU computation."""
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    class_name: str = "object"
    confidence: float = 1.0
    track_id: Optional[str] = None

    def compute_iou(self, other: "BBox2D") -> float:
        """Computes standard Intersection over Union (IoU) between two bounding boxes."""
        inter_xmin = max(self.xmin, other.xmin)
        inter_ymin = max(self.ymin, other.ymin)
        inter_xmax = min(self.xmax, other.xmax)
        inter_ymax = min(self.ymax, other.ymax)

        inter_w = max(0.0, inter_xmax - inter_xmin)
        inter_h = max(0.0, inter_ymax - inter_ymin)
        inter_area = inter_w * inter_h

        area_self = max(0.0, self.xmax - self.xmin) * max(0.0, self.ymax - self.ymin)
        area_other = max(0.0, other.xmax - other.xmin) * max(0.0, other.ymax - other.ymin)
        union_area = area_self + area_other - inter_area

        if union_area <= 1e-6:
            return 0.0
        return float(inter_area / union_area)


@dataclass
class MetricScores:
    """Quantitative metric scores evaluating structural and semantic consistency."""
    mean_bbox_iou: float
    object_count_drift: int
    edge_similarity: float           # Edge intersection over union
    edge_preservation_recall: float  # Fraction of original structural edges preserved
    semantic_mask_iou: Optional[float] = None
    depth_correlation: Optional[float] = None
    fid_batch_score: Optional[float] = None  # Distributional proxy on batches only


@dataclass
class QCReport:
    """Comprehensive Quality Gate verification report."""
    status: QCStatus
    metrics: MetricScores
    passed_checks: Dict[str, bool]
    failure_reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "passed_checks": self.passed_checks,
            "failure_reasons": self.failure_reasons,
            "metrics": {
                "mean_bbox_iou": self.metrics.mean_bbox_iou,
                "object_count_drift": self.metrics.object_count_drift,
                "edge_similarity": self.metrics.edge_similarity,
                "edge_preservation_recall": self.metrics.edge_preservation_recall,
                "semantic_mask_iou": self.metrics.semantic_mask_iou,
                "depth_correlation": self.metrics.depth_correlation,
                "fid_batch_score": self.metrics.fid_batch_score
            },
            "metadata": self.metadata
        }
