"""Unit tests for Phase 6 downstream export and benchmark suite."""

import json
from pathlib import Path
import numpy as np
import pytest
from evaluation import BBox2D, DatasetExporter, DownstreamBenchmark


def test_coco_and_yolo_export(tmp_path):
    exporter = DatasetExporter(str(tmp_path))

    # Mock samples
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    samples = [
        {
            "id": "frame_001",
            "image_array": img,
            "width": 640,
            "height": 480,
            "boxes": [
                BBox2D(xmin=100, ymin=100, xmax=200, ymax=200, class_name="car"),
                BBox2D(xmin=300, ymin=250, xmax=350, ymax=350, class_name="person")
            ]
        },
        {
            "id": "frame_002",
            "image_array": img,
            "width": 640,
            "height": 480,
            "boxes": [
                BBox2D(xmin=50, ymin=50, xmax=150, ymax=150, class_name="building")
            ]
        }
    ]

    # Test COCO export
    coco_json = exporter.export_coco(samples, split_name="test_split")
    assert coco_json.exists()
    with open(coco_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data["images"]) == 2
    assert len(data["annotations"]) == 3
    assert (tmp_path / "test_split" / "images" / "frame_001.png").exists()

    # Test YOLO export
    yolo_dir = exporter.export_yolo(samples, split_name="test_yolo")
    assert (yolo_dir / "images" / "frame_001.png").exists()
    lbl_file = yolo_dir / "labels" / "frame_001.txt"
    assert lbl_file.exists()
    with open(lbl_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 2


def test_downstream_benchmark_utility():
    bench = DownstreamBenchmark()

    gt = [
        [BBox2D(xmin=10, ymin=10, xmax=50, ymax=50, class_name="car")],
        [BBox2D(xmin=20, ymin=20, xmax=60, ymax=60, class_name="person")]
    ]

    # Baseline detector missed sample 2
    base_preds = [
        [BBox2D(xmin=10, ymin=10, xmax=50, ymax=50, class_name="car")],
        []
    ]

    # Augmented detector detected both accurately
    aug_preds = [
        [BBox2D(xmin=10, ymin=10, xmax=50, ymax=50, class_name="car")],
        [BBox2D(xmin=20, ymin=20, xmax=60, ymax=60, class_name="person")]
    ]

    result = bench.evaluate_detector_gain(gt, base_preds, aug_preds)
    assert result.baseline_map50 == 0.5
    assert result.augmented_map50 == 1.0
    assert result.delta_gain == 0.5
    assert result.summary_report["decision"] == "UPLIFT_CONFIRMED"
