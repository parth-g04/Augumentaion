from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


QCStatus = Literal["PASS", "WARNING", "REJECT"]


@dataclass
class BBox2D:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    class_name: str = "object"
    confidence: float = 1.0

    def compute_iou(self, other: "BBox2D") -> float:
        x1 = max(self.xmin, other.xmin)
        y1 = max(self.ymin, other.ymin)
        x2 = min(self.xmax, other.xmax)
        y2 = min(self.ymax, other.ymax)

        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        intersection = inter_w * inter_h

        area_a = max(0.0, self.xmax - self.xmin) * max(
            0.0, self.ymax - self.ymin
        )
        area_b = max(0.0, other.xmax - other.xmin) * max(
            0.0, other.ymax - other.ymin
        )

        union = area_a + area_b - intersection

        if union <= 0:
            return 0.0

        return float(intersection / union)


@dataclass
class MetricScores:
    mean_bbox_iou: Optional[float] = None
    object_count_drift: Optional[int] = None

    edge_similarity: Optional[float] = None
    edge_preservation_recall: Optional[float] = None
    edge_chance_corrected: Optional[float] = None

    semantic_mask_iou: Optional[float] = None
    depth_correlation: Optional[float] = None

    image_size_match: Optional[bool] = None
    registration_used: bool = False


@dataclass
class QCReport:
    status: QCStatus
    metrics: MetricScores
    passed_checks: Dict[str, Optional[bool]]
    failure_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "metrics": {
                "mean_bbox_iou": self.metrics.mean_bbox_iou,
                "object_count_drift": self.metrics.object_count_drift,
                "edge_similarity": self.metrics.edge_similarity,
                "edge_preservation_recall":
                    self.metrics.edge_preservation_recall,
                "edge_chance_corrected":
                    self.metrics.edge_chance_corrected,
                "semantic_mask_iou":
                    self.metrics.semantic_mask_iou,
                "depth_correlation":
                    self.metrics.depth_correlation,
                "image_size_match":
                    self.metrics.image_size_match,
                "registration_used":
                    self.metrics.registration_used,
            },
            "passed_checks": self.passed_checks,
            "failure_reasons": self.failure_reasons,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }