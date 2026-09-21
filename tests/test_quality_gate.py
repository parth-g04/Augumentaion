"""Unit tests for Phase 4 Quality Gate and Consistency Checker."""

import numpy as np
import pytest
from evaluation import BBox2D, QualityGate, QualityGateThresholds


def test_bbox_iou_computation():
    b1 = BBox2D(xmin=10, ymin=10, xmax=50, ymax=50)
    b2 = BBox2D(xmin=10, ymin=10, xmax=50, ymax=50)
    # Identical boxes
    assert b1.compute_iou(b2) == 1.0

    # Non-overlapping boxes
    b3 = BBox2D(xmin=60, ymin=60, xmax=100, ymax=100)
    assert b1.compute_iou(b3) == 0.0

    # Partial overlap (half width)
    b4 = BBox2D(xmin=30, ymin=10, xmax=50, ymax=50)
    # Intersection = 20*40 = 800, Union = 40*40 + 20*40 - 800 = 1600. IoU = 0.5
    assert np.isclose(b1.compute_iou(b4), 0.5)


def test_quality_gate_pass_on_compliant_augmentation():
    gate = QualityGate()

    orig_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    orig_rgb[20:80, 20:80] = 200

    # Slightly modified image (e.g., rain or slight sensor noise) preserving structural edges
    aug_rgb = orig_rgb.copy()
    aug_rgb[20:80, 20:80] = 180

    orig_boxes = [
        BBox2D(xmin=20, ymin=20, xmax=50, ymax=50, class_name="car"),
        BBox2D(xmin=55, ymin=55, xmax=80, ymax=80, class_name="person")
    ]
    # Augmented boxes with tiny subpixel jitter (IoU > 0.9)
    aug_boxes = [
        BBox2D(xmin=21, ymin=20, xmax=51, ymax=50, class_name="car"),
        BBox2D(xmin=55, ymin=54, xmax=80, ymax=81, class_name="person")
    ]

    report = gate.evaluate(orig_rgb, aug_rgb, orig_boxes, aug_boxes)
    assert report.status == "PASS"
    assert report.metrics.object_count_drift == 0
    assert report.metrics.mean_bbox_iou > 0.9
    assert len(report.failure_reasons) == 0


def test_quality_gate_reject_on_drifted_or_hallucinated_image():
    # Strict thresholds
    thresholds = QualityGateThresholds(min_bbox_iou=0.70, max_count_drift=1, min_edge_similarity=0.50)
    gate = QualityGate(thresholds)

    orig_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    orig_rgb[20:80, 20:80] = 255

    # Completely blurred/melted image (no edges preserved)
    aug_rgb = np.full((100, 100, 3), 128, dtype=np.uint8)

    orig_boxes = [
        BBox2D(xmin=10, ymin=10, xmax=40, ymax=40, class_name="car"),
        BBox2D(xmin=50, ymin=50, xmax=80, ymax=80, class_name="truck"),
        BBox2D(xmin=82, ymin=82, xmax=95, ymax=95, class_name="person")
    ]
    # Boxes completely missing or hallucinated (count drift = 3, IoU = 0)
    aug_boxes = []

    report = gate.evaluate(orig_rgb, aug_rgb, orig_boxes, aug_boxes)
    assert report.status == "REJECT"
    assert "count_drift_within_tolerance" in report.passed_checks
    assert report.passed_checks["count_drift_within_tolerance"] is False
    assert len(report.failure_reasons) > 0
