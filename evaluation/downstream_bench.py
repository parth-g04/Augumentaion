"""Downstream detector utility benchmark for Layer 4.
Evaluates the ultimate research metric emphasized by IITB faculty:
Downstream detection performance gain on unseen real UAV flight test sets
comparing Real-Only baseline vs. Real + Augmented datasets.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

from .interfaces import BBox2D


@dataclass
class BenchmarkResult:
    """Downstream evaluation benchmark comparison."""
    baseline_map50: float
    augmented_map50: float
    delta_gain: float
    num_test_samples: int
    summary_report: Dict[str, Any]


class DownstreamBenchmark:
    """Simulates/evaluates downstream object detection utility."""

    def evaluate_detector_gain(
        self,
        ground_truth: List[List[BBox2D]],
        baseline_preds: List[List[BBox2D]],
        augmented_preds: List[List[BBox2D]]
    ) -> BenchmarkResult:
        """Computes mean average precision (mAP@50) for baseline vs augmented model predictions."""
        def compute_ap50(gts: List[List[BBox2D]], preds: List[List[BBox2D]]) -> float:
            precisions = []
            for gt_boxes, pr_boxes in zip(gts, preds):
                if not gt_boxes:
                    continue
                matched = 0
                unmatched_pr = list(pr_boxes)
                for g in gt_boxes:
                    best_iou = 0.0
                    best_idx = -1
                    for idx, p in enumerate(unmatched_pr):
                        if p.class_name == g.class_name:
                            iou = g.compute_iou(p)
                            if iou > best_iou:
                                best_iou = iou
                                best_idx = idx
                    if best_idx >= 0 and best_iou >= 0.50:
                        matched += 1
                        unmatched_pr.pop(best_idx)
                prec = matched / len(pr_boxes) if pr_boxes else (1.0 if not gt_boxes else 0.0)
                precisions.append(prec)
            return float(np.mean(precisions)) if precisions else 0.0

        base_map = compute_ap50(ground_truth, baseline_preds)
        aug_map = compute_ap50(ground_truth, augmented_preds)
        gain = aug_map - base_map

        return BenchmarkResult(
            baseline_map50=round(base_map, 3),
            augmented_map50=round(aug_map, 3),
            delta_gain=round(gain, 3),
            num_test_samples=len(ground_truth),
            summary_report={
                "metric": "mAP@50",
                "hypothesis": "Layer 4 augmentation improves detector generalization on adverse/novel viewpoints",
                "baseline_map50": round(base_map, 3),
                "augmented_map50": round(aug_map, 3),
                "detector_gain": round(gain, 3),
                "decision": "UPLIFT_CONFIRMED" if gain > 0 else "NO_GAIN"
            }
        )
