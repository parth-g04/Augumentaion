"""
test_quality_report.py
-----------------------
Runs the full evaluation/ QC pipeline on a real ORIGINAL image and a real
GENERATED (weather-augmented) image, and produces a QC report.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from evaluation.interfaces import BBox2D
from evaluation.quality_gate import QualityGate, QualityGateThresholds


def load_image(path: str, resize_to=None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if resize_to is not None:
        img = img.resize(resize_to)
    return np.array(img)


def load_boxes(path):
    if path is None:
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [
        BBox2D(
            xmin=b["xmin"], ymin=b["ymin"],
            xmax=b["xmax"], ymax=b["ymax"],
            class_name=b.get("class_name", "object"),
            confidence=b.get("confidence", 1.0),
        )
        for b in raw
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True)
    parser.add_argument("--generated", required=True)
    parser.add_argument("--orig-boxes", default=None)
    parser.add_argument("--gen-boxes", default=None)
    parser.add_argument("--out", default="qc_report.json")
    parser.add_argument("--min-bbox-iou", type=float, default=0.5)
    parser.add_argument("--max-count-drift", type=int, default=1)
    parser.add_argument("--min-edge-similarity", type=float, default=0.15)
    args = parser.parse_args()

    orig_rgb = load_image(args.original)
    gen_rgb = load_image(args.generated, resize_to=(orig_rgb.shape[1], orig_rgb.shape[0]))
    orig_boxes = load_boxes(args.orig_boxes)
    gen_boxes = load_boxes(args.gen_boxes)

    thresholds = QualityGateThresholds(
        min_bbox_iou=args.min_bbox_iou,
        max_count_drift=args.max_count_drift,
        min_edge_similarity=args.min_edge_similarity,
    )

    gate = QualityGate(thresholds)
    report = gate.evaluate(
        orig_rgb=orig_rgb,
        aug_rgb=gen_rgb,
        orig_boxes=orig_boxes,
        aug_boxes=gen_boxes,
        metadata={
            "original_path": args.original,
            "generated_path": args.generated,
            "had_box_annotations": bool(orig_boxes or gen_boxes),
        },
    )

    report_dict = report.to_dict()
    print(json.dumps(report_dict, indent=2))

    Path(args.out).write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
    print(f"\nSaved report to {args.out}")


if __name__ == "__main__":
    main()
