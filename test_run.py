"""
test_quality_report.py
-----------------------
Runs the full evaluation/ QC pipeline on a real ORIGINAL image and a real
GENERATED (weather-augmented) image, and produces a QC report.

Place this at your repo root, sibling to evaluation/ (same rule as
test_run.py and run_qc_on_real_data.py).

For scenes with NO object-level bounding boxes (e.g. a 3DGS scene render
with no detection annotations), pass no --orig-boxes/--gen-boxes and the
gate will run with empty box lists. In that case:
    - bbox_consistency and count_drift_within_tolerance both trivially
      PASS (nothing to check against), which is correct behavior — not
      a false positive, just "not applicable here".
    - structural_edge_preserved becomes the metric that actually matters
      for this kind of scene: it's checking whether the fog/weather
      conditioning preserved the underlying building/road/vegetation
      geometry rather than hallucinating or smearing it away.

If you DO have boxes for a scene, pass --orig-boxes/--gen-boxes as JSON
files in the same format used by run_qc_on_real_data.py.

Usage (no boxes — geometry-only check):
    python test_quality_report.py \\
        --original path/to/original.png \\
        --generated path/to/generated_fog.png \\
        --out qc_report.json

Usage (with boxes, if you have them for this scene):
    python test_quality_report.py \\
        --original path/to/original.png \\
        --generated path/to/generated_fog.png \\
        --orig-boxes path/to/original_boxes.json \\
        --gen-boxes path/to/generated_boxes.json \\
        --out qc_report.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from evaluation.interfaces import BBox2D
from evaluation.quality_gate import QualityGate, QualityGateThresholds


def load_image(path: str, resize_to: tuple[int, int] | None = None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if resize_to is not None:
        img = img.resize(resize_to)
    return np.array(img)


def load_boxes(path: str | None) -> list[BBox2D]:
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
    parser.add_argument("--original", required=True, help="Path to original/reference render")
    parser.add_argument("--generated", required=True, help="Path to generated (weather-augmented) image")
    parser.add_argument("--orig-boxes", default=None, help="Optional GT boxes JSON")
    parser.add_argument("--gen-boxes", default=None, help="Optional detected boxes JSON")
    parser.add_argument("--out", default="qc_report.json")

    # Thresholds — relaxed edge-similarity default, since heavy weather
    # conditioning (like dense fog) is EXPECTED to reduce raw edge overlap
    # even when structure is well-preserved. Tighten these once you've
    # run this across several samples and see your actual score range.
    parser.add_argument("--min-bbox-iou", type=float, default=0.5)
    parser.add_argument("--max-count-drift", type=int, default=1)
    parser.add_argument("--min-edge-similarity", type=float, default=0.15)
    args = parser.parse_args()

    orig_rgb = load_image(args.original)
    # Original and generated renders can come out at slightly different
    # pixel dimensions (off-by-a-few-pixel rounding from the render/export
    # step) — resize the generated image to match the original before any
    # metric computation, since edge_similarity.py assumes equal shapes.
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