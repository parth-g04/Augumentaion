"""
LAYER 4 - QUALITY CONTROL / EVALUATION

Checks:
    1. Image size
    2. Structural edge preservation
    3. Bounding-box IoU
    4. Hard per-object IoU
    5. New object detection
    6. Object count drift
    7. Semantic IoU
    8. Dataset-level FID

Important:
    - Missing annotations are NOT treated as passing.
    - Missing metrics are shown as N/E (Not Evaluated).
    - Semantic IoU threshold is TBD by default.
    - FID is a dataset-level metric.
    - The final report is always saved to qc_report.json.
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


# ============================================================
# DEFAULT THRESHOLDS
# ============================================================

DEFAULT_EDGE_THRESHOLD = 0.15
DEFAULT_BBOX_THRESHOLD = 0.60
DEFAULT_COUNT_DRIFT = 2
DEFAULT_FID_THRESHOLD = 15.0

# Project requirement says semantic IoU should be
# determined empirically.
DEFAULT_SEMANTIC_THRESHOLD = None


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(path: str) -> np.ndarray:
    """Load image as RGB."""

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Image not found: {path}"
        )

    image = cv2.imread(path)

    if image is None:
        raise ValueError(
            f"Could not read image: {path}"
        )

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


# ============================================================
# IMAGE SIZE / RESIZING
# ============================================================

def resize_to_reference(
    original: np.ndarray,
    generated: np.ndarray
):
    """
    Resize generated image to original dimensions.

    Returns:
        resized_generated
        image_info
    """

    orig_h, orig_w = original.shape[:2]
    gen_h, gen_w = generated.shape[:2]

    original_ratio = orig_w / orig_h
    generated_ratio = gen_w / gen_h

    if (
        orig_h == gen_h
        and orig_w == gen_w
    ):
        return generated, {
            "resized": False,
            "original_size": [
                orig_w,
                orig_h
            ],
            "generated_size": [
                gen_w,
                gen_h
            ],
            "aspect_ratio_original": round(
                original_ratio,
                4
            ),
            "aspect_ratio_generated": round(
                generated_ratio,
                4
            ),
            "aspect_ratio_difference": round(
                abs(
                    original_ratio
                    - generated_ratio
                ),
                4
            )
        }

    resized = cv2.resize(
        generated,
        (
            orig_w,
            orig_h
        ),
        interpolation=cv2.INTER_AREA
    )

    return resized, {
        "resized": True,
        "original_size": [
            orig_w,
            orig_h
        ],
        "generated_size": [
            gen_w,
            gen_h
        ],
        "aspect_ratio_original": round(
            original_ratio,
            4
        ),
        "aspect_ratio_generated": round(
            generated_ratio,
            4
        ),
        "aspect_ratio_difference": round(
            abs(
                original_ratio
                - generated_ratio
            ),
            4
        )
    }


# ============================================================
# ROBUST EDGE DETECTION
# ============================================================

def get_edges_robust(
    image: np.ndarray,
    keep_fraction: float = 0.08
) -> np.ndarray:
    """
    Extract structural edges using:

        RGB -> luminance
        luminance -> log transform
        Sobel gradient
        strongest gradient pixels

    This is the robust method used by the earlier
    successful QC calculation.
    """

    image = (
        image.astype(np.float32)
        / 255.0
    )

    # RGB -> luminance
    gray = (
        0.299 * image[..., 0]
        + 0.587 * image[..., 1]
        + 0.114 * image[..., 2]
    )

    # Log-luminance
    gray = np.log1p(
        gray * 10.0
    )

    # Horizontal gradient
    gx = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    # Vertical gradient
    gy = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    # Gradient magnitude
    gradient = np.sqrt(
        gx ** 2
        + gy ** 2
    )

    flat = gradient.flatten()

    if len(flat) == 0:
        return np.zeros_like(
            gradient,
            dtype=bool
        )

    # Keep strongest gradients
    keep_fraction = max(
        0.001,
        min(
            keep_fraction,
            1.0
        )
    )

    threshold = np.quantile(
        flat,
        1.0 - keep_fraction
    )

    return (
        gradient >= threshold
    )


def compute_edge_metrics_robust(
    original: np.ndarray,
    generated: np.ndarray,
    keep_fraction: float = 0.08,
    tolerance: int = 2
) -> Dict[str, float]:
    """
    Calculate the robust structural edge score.

    Metrics:

        edge_recall_tolerant
            How much of the original structure
            is found in the generated image.

        chance_level
            Expected overlap by chance.

        edge_recall_chance_corrected
            Final structural score used by
            the Layer 4 quality gate.

        lit_fraction
            Fraction of original image considered
            sufficiently illuminated.
    """

    original_edges = get_edges_robust(
        original,
        keep_fraction
    )

    generated_edges = get_edges_robust(
        generated,
        keep_fraction
    )

    # --------------------------------------------------------
    # Tolerant matching
    # --------------------------------------------------------

    kernel_size = (
        2 * tolerance + 1
    )

    kernel = np.ones(
        (
            kernel_size,
            kernel_size
        ),
        dtype=np.uint8
    )

    generated_dilated = cv2.dilate(
        generated_edges.astype(
            np.uint8
        ),
        kernel
    ).astype(bool)

    original_count = (
        original_edges.sum()
    )

    if original_count > 0:

        matched = np.logical_and(
            original_edges,
            generated_dilated
        ).sum()

        edge_recall_tolerant = (
            matched
            / original_count
        )

    else:

        edge_recall_tolerant = 1.0

    # --------------------------------------------------------
    # Chance level
    # --------------------------------------------------------

    original_fraction = float(
        original_edges.mean()
    )

    generated_fraction = float(
        generated_edges.mean()
    )

    chance_level = (
        original_fraction
        + generated_fraction
        - (
            original_fraction
            * generated_fraction
        )
    )

    # --------------------------------------------------------
    # Chance-corrected score
    # --------------------------------------------------------

    denominator = (
        1.0 - chance_level
    )

    if denominator > 1e-8:

        chance_corrected = (
            edge_recall_tolerant
            - chance_level
        ) / denominator

    else:

        chance_corrected = 0.0

    chance_corrected = max(
        0.0,
        min(
            1.0,
            chance_corrected
        )
    )

    # --------------------------------------------------------
    # Lighting information
    # --------------------------------------------------------

    original_gray = (
        0.299 * original[..., 0]
        + 0.587 * original[..., 1]
        + 0.114 * original[..., 2]
    )

    lit_fraction = float(
        np.mean(
            original_gray > 20
        )
    )

    return {
        "edge_recall_tolerant": float(
            edge_recall_tolerant
        ),
        "chance_level": float(
            chance_level
        ),
        "edge_recall_chance_corrected": float(
            chance_corrected
        ),
        "lit_fraction": float(
            lit_fraction
        )
    }


# ============================================================
# BOUNDING BOX CLASS
# ============================================================

class SimpleBBox:

    def __init__(
        self,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
        class_name: str = "unknown",
        confidence: float = 1.0
    ):

        self.xmin = float(xmin)
        self.ymin = float(ymin)
        self.xmax = float(xmax)
        self.ymax = float(ymax)

        self.class_name = class_name
        self.confidence = confidence

    def compute_iou(
        self,
        other
    ) -> float:

        inter_xmin = max(
            self.xmin,
            other.xmin
        )

        inter_ymin = max(
            self.ymin,
            other.ymin
        )

        inter_xmax = min(
            self.xmax,
            other.xmax
        )

        inter_ymax = min(
            self.ymax,
            other.ymax
        )

        inter_width = max(
            0.0,
            inter_xmax
            - inter_xmin
        )

        inter_height = max(
            0.0,
            inter_ymax
            - inter_ymin
        )

        intersection = (
            inter_width
            * inter_height
        )

        area_a = (
            max(
                0.0,
                self.xmax
                - self.xmin
            )
            *
            max(
                0.0,
                self.ymax
                - self.ymin
            )
        )

        area_b = (
            max(
                0.0,
                other.xmax
                - other.xmin
            )
            *
            max(
                0.0,
                other.ymax
                - other.ymin
            )
        )

        union = (
            area_a
            + area_b
            - intersection
        )

        if union <= 0:
            return 0.0

        return float(
            intersection / union
        )

    def center(self):

        return (
            (
                self.xmin
                + self.xmax
            ) / 2.0,

            (
                self.ymin
                + self.ymax
            ) / 2.0
        )


# ============================================================
# LOAD BOUNDING BOXES
# ============================================================

def load_boxes(
    path: str
) -> List[SimpleBBox]:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Bounding-box file not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    if not isinstance(
        data,
        list
    ):

        raise ValueError(
            "Bounding-box JSON must "
            "contain a list."
        )

    boxes = []

    for item in data:

        boxes.append(
            SimpleBBox(
                xmin=item["xmin"],
                ymin=item["ymin"],
                xmax=item["xmax"],
                ymax=item["ymax"],
                class_name=item.get(
                    "class_name",
                    "unknown"
                ),
                confidence=item.get(
                    "confidence",
                    1.0
                )
            )
        )

    return boxes


# ============================================================
# HARD PER-OBJECT BBOX CHECK
# ============================================================

def evaluate_bounding_boxes(
    original_boxes: List[SimpleBBox],
    generated_boxes: List[SimpleBBox],
    iou_threshold: float
) -> Dict[str, Any]:
    """
    Hard object-level check.

    PASS requires:

        1. Every original object has a match.
        2. Every matched object has IoU >= threshold.
        3. No new generated objects exist.
    """

    unused_generated = list(
        generated_boxes
    )

    object_results = []

    for index, original in enumerate(
        original_boxes
    ):

        best_iou = 0.0
        best_index = -1

        for j, generated in enumerate(
            unused_generated
        ):

            if (
                generated.class_name
                != original.class_name
            ):
                continue

            iou = original.compute_iou(
                generated
            )

            if iou > best_iou:

                best_iou = iou
                best_index = j

        if best_index >= 0:

            matched = unused_generated.pop(
                best_index
            )

            orig_cx, orig_cy = (
                original.center()
            )

            gen_cx, gen_cy = (
                matched.center()
            )

            displacement = float(
                np.sqrt(
                    (
                        orig_cx
                        - gen_cx
                    ) ** 2
                    +
                    (
                        orig_cy
                        - gen_cy
                    ) ** 2
                )
            )

            passed = (
                best_iou
                >= iou_threshold
            )

            object_results.append(
                {
                    "object_index": (
                        index + 1
                    ),
                    "class_name": (
                        original.class_name
                    ),
                    "iou": round(
                        best_iou,
                        4
                    ),
                    "center_displacement": round(
                        displacement,
                        2
                    ),
                    "passed": passed
                }
            )

        else:

            object_results.append(
                {
                    "object_index": (
                        index + 1
                    ),
                    "class_name": (
                        original.class_name
                    ),
                    "iou": 0.0,
                    "center_displacement": None,
                    "passed": False,
                    "reason": (
                        "No matching "
                        "generated object"
                    )
                }
            )

    # Remaining generated boxes are new objects.
    new_objects = []

    for generated in unused_generated:

        new_objects.append(
            {
                "class_name": (
                    generated.class_name
                ),
                "xmin": generated.xmin,
                "ymin": generated.ymin,
                "xmax": generated.xmax,
                "ymax": generated.ymax
            }
        )

    if object_results:

        mean_iou = float(
            np.mean(
                [
                    obj["iou"]
                    for obj
                    in object_results
                ]
            )
        )

    else:

        mean_iou = None

    all_objects_pass = all(
        obj["passed"]
        for obj
        in object_results
    )

    no_new_objects = (
        len(new_objects) == 0
    )

    hard_pass = (
        all_objects_pass
        and no_new_objects
    )

    return {
        "mean_iou": (
            round(
                mean_iou,
                4
            )
            if mean_iou is not None
            else None
        ),
        "original_object_count": (
            len(original_boxes)
        ),
        "generated_object_count": (
            len(generated_boxes)
        ),
        "count_drift": abs(
            len(original_boxes)
            - len(generated_boxes)
        ),
        "objects": object_results,
        "new_objects": new_objects,
        "new_object_count": len(
            new_objects
        ),
        "all_objects_pass": (
            all_objects_pass
        ),
        "no_new_objects": (
            no_new_objects
        ),
        "hard_pass": hard_pass,
        "threshold": iou_threshold
    }


# ============================================================
# SEMANTIC MASK
# ============================================================

def load_mask(
    path: str
) -> np.ndarray:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Mask not found: {path}"
        )

    mask = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:

        raise ValueError(
            f"Could not read mask: {path}"
        )

    return mask


def compute_semantic_iou(
    original_mask: np.ndarray,
    generated_mask: np.ndarray
) -> float:

    if (
        original_mask.shape
        != generated_mask.shape
    ):

        generated_mask = cv2.resize(
            generated_mask,
            (
                original_mask.shape[1],
                original_mask.shape[0]
            ),
            interpolation=cv2.INTER_NEAREST
        )

    original_foreground = (
        original_mask > 0
    )

    generated_foreground = (
        generated_mask > 0
    )

    intersection = np.logical_and(
        original_foreground,
        generated_foreground
    ).sum()

    union = np.logical_or(
        original_foreground,
        generated_foreground
    ).sum()

    if union == 0:
        return 1.0

    return float(
        intersection / union
    )


# ============================================================
# FID
# ============================================================

def compute_fid_if_requested(
    real_dir: Optional[str],
    generated_dir: Optional[str]
) -> Dict[str, Any]:

    if (
        not real_dir
        or not generated_dir
    ):

        return {
            "evaluated": False,
            "score": None,
            "reason": (
                "Not evaluated — "
                "dataset folders were "
                "not provided."
            )
        }

    try:

        from evaluation.fid_metric import (
            compute_fid
        )

        score, real_count, generated_count = (
            compute_fid(
                real_dir,
                generated_dir
            )
        )

        return {
            "evaluated": True,
            "score": round(
                score,
                4
            ),
            "real_image_count": (
                real_count
            ),
            "generated_image_count": (
                generated_count
            ),
            "reason": None
        }

    except ImportError as e:

        return {
            "evaluated": False,
            "score": None,
            "reason": (
                "FID dependencies are "
                "not installed. "
                f"Details: {e}"
            )
        }

    except Exception as e:

        return {
            "evaluated": False,
            "score": None,
            "reason": (
                f"FID calculation failed: {e}"
            )
        }


# ============================================================
# METRIC EVALUATION
# ============================================================

def evaluate_metric(
    actual: Optional[float],
    threshold: Optional[float],
    comparison: str
):

    if actual is None:
        return "N/E"

    if threshold is None:
        return "TBD"

    if comparison == ">=":

        return (
            "PASS"
            if actual >= threshold
            else "FAIL"
        )

    if comparison == "<=":

        return (
            "PASS"
            if actual <= threshold
            else "FAIL"
        )

    if comparison == "<":

        return (
            "PASS"
            if actual < threshold
            else "FAIL"
        )

    return "N/E"


def determine_final_status(
    results: List[str]
) -> str:

    evaluated = [
        result
        for result in results
        if result not in (
            "N/E",
            "TBD"
        )
    ]

    if any(
        result == "FAIL"
        for result in evaluated
    ):

        return "REJECT"

    if any(
        result in (
            "N/E",
            "TBD"
        )
        for result in results
    ):

        return "WARNING"

    return "PASS"


# ============================================================
# PRINT REPORT
# ============================================================

def print_report(
    original_path,
    generated_path,
    status,
    image_info,
    edge_metrics,
    bbox_result,
    semantic_iou,
    fid_result,
    edge_threshold,
    bbox_threshold,
    count_threshold,
    semantic_threshold,
    fid_threshold
):

    print()
    print("=" * 78)
    print(
        "                         LAYER 4 QUALITY REPORT"
    )
    print("=" * 78)

    print()

    print(
        f"Original  : "
        f"{os.path.basename(original_path)}"
    )

    print(
        f"Generated : "
        f"{os.path.basename(generated_path)}"
    )

    print()

    print(
        f"STATUS : {status}"
    )

    print()

    print("-" * 78)

    print(
        f"{'Metric':<25}"
        f"{'Actual':<15}"
        f"{'Threshold':<17}"
        f"{'Result'}"
    )

    print("-" * 78)

    # --------------------------------------------------------
    # Image size
    # --------------------------------------------------------

    image_result = (
        "PASS"
        if not image_info["resized"]
        else "WARNING"
    )

    image_actual = (
        "MATCH"
        if not image_info["resized"]
        else "RESIZED"
    )

    print(
        f"{'Image Size':<25}"
        f"{image_actual:<15}"
        f"{'Same size':<17}"
        f"{image_result}"
    )

    # --------------------------------------------------------
    # EDGE
    # --------------------------------------------------------

    edge_score = (
        edge_metrics[
            "edge_recall_chance_corrected"
        ]
    )

    edge_result = evaluate_metric(
        edge_score,
        edge_threshold,
        ">="
    )

    print(
        f"{'Structural Edge Score':<25}"
        f"{edge_score:<15.3f}"
        f"{'>= ' + format(edge_threshold, '.2f'):<17}"
        f"{edge_result}"
    )

    # --------------------------------------------------------
    # BBOX
    # --------------------------------------------------------

    if bbox_result is None:

        print(
            f"{'BBox IoU':<25}"
            f"{'N/E':<15}"
            f"{'>= ' + format(bbox_threshold, '.2f'):<17}"
            f"N/E"
        )

        print(
            f"{'Per-object IoU':<25}"
            f"{'N/E':<15}"
            f"{'>= ' + format(bbox_threshold, '.2f'):<17}"
            f"N/E"
        )

        print(
            f"{'Object Count Drift':<25}"
            f"{'N/E':<15}"
            f"{'<= ' + str(count_threshold):<17}"
            f"N/E"
        )

    else:

        mean_iou = (
            bbox_result["mean_iou"]
        )

        bbox_result_status = evaluate_metric(
            mean_iou,
            bbox_threshold,
            ">="
        )

        print(
            f"{'BBox IoU':<25}"
            f"{mean_iou:<15.3f}"
            f"{'>= ' + format(bbox_threshold, '.2f'):<17}"
            f"{bbox_result_status}"
        )

        hard_status = (
            "PASS"
            if bbox_result["hard_pass"]
            else "FAIL"
        )

        print(
            f"{'Per-object IoU':<25}"
            f"{hard_status:<15}"
            f"{'>= ' + format(bbox_threshold, '.2f'):<17}"
            f"{hard_status}"
        )

        count_drift = (
            bbox_result["count_drift"]
        )

        count_status = evaluate_metric(
            count_drift,
            count_threshold,
            "<="
        )

        print(
            f"{'Object Count Drift':<25}"
            f"{count_drift:<15}"
            f"{'<= ' + str(count_threshold):<17}"
            f"{count_status}"
        )

    # --------------------------------------------------------
    # SEMANTIC IoU
    # --------------------------------------------------------

    if semantic_iou is None:

        semantic_actual = "N/E"
        semantic_threshold_text = "TBD"
        semantic_result = "N/E"

    else:

        semantic_actual = (
            f"{semantic_iou:.3f}"
        )

        if semantic_threshold is None:

            semantic_threshold_text = "TBD"
            semantic_result = "TBD"

        else:

            semantic_threshold_text = (
                f">= {semantic_threshold:.2f}"
            )

            semantic_result = evaluate_metric(
                semantic_iou,
                semantic_threshold,
                ">="
            )

    print(
        f"{'Semantic IoU':<25}"
        f"{semantic_actual:<15}"
        f"{semantic_threshold_text:<17}"
        f"{semantic_result}"
    )

    # --------------------------------------------------------
    # FID
    # --------------------------------------------------------

    if fid_result["score"] is None:

        print(
            f"{'FID':<25}"
            f"{'N/E':<15}"
            f"{'< ' + format(fid_threshold, '.2f'):<17}"
            f"N/E"
        )

    else:

        fid_score = (
            fid_result["score"]
        )

        fid_status = evaluate_metric(
            fid_score,
            fid_threshold,
            "<"
        )

        print(
            f"{'FID':<25}"
            f"{fid_score:<15.3f}"
            f"{'< ' + format(fid_threshold, '.2f'):<17}"
            f"{fid_status}"
        )

    print("-" * 78)

    # ========================================================
    # REASONS
    # ========================================================

    print()
    print("REASONS")
    print()

    # Edge
    if edge_result == "PASS":

        print(
            f"  Structural edges : "
            f"{edge_score:.4f} >= "
            f"{edge_threshold:.4f} -> PASS"
        )

    else:

        print(
            f"  Structural edges : "
            f"{edge_score:.4f} < "
            f"{edge_threshold:.4f} -> FAIL"
        )

    # BBox
    if bbox_result is None:

        print(
            "  BBox             : "
            "Not evaluated — bounding-box "
            "annotations were not provided."
        )

        print(
            "  Object Count     : "
            "Not evaluated — bounding-box "
            "annotations were not provided."
        )

    else:

        if bbox_result["hard_pass"]:

            print(
                "  BBox             : "
                "All objects satisfy the hard "
                f"IoU threshold >= "
                f"{bbox_threshold:.2f} "
                "and no new objects were detected."
            )

        else:

            print(
                "  BBox             : "
                "Hard per-object constraint FAILED."
            )

            failed_objects = [
                obj
                for obj in bbox_result["objects"]
                if not obj["passed"]
            ]

            for obj in failed_objects:

                print(
                    f"    Object "
                    f"{obj['object_index']} "
                    f"({obj['class_name']}): "
                    f"IoU = "
                    f"{obj['iou']:.3f}"
                )

            if (
                bbox_result[
                    "new_object_count"
                ] > 0
            ):

                print(
                    "    New objects detected: "
                    f"{bbox_result['new_object_count']}"
                )

        print(
            f"  Object Count     : "
            f"drift = "
            f"{bbox_result['count_drift']}, "
            f"allowed = "
            f"{count_threshold}"
        )

    # Semantic
    if semantic_iou is None:

        print(
            "  Semantic IoU     : "
            "Not evaluated — segmentation "
            "masks were not provided."
        )

    elif semantic_threshold is None:

        print(
            f"  Semantic IoU     : "
            f"{semantic_iou:.3f} — "
            "threshold is TBD and must be "
            "validated empirically."
        )

    elif semantic_iou >= semantic_threshold:

        print(
            f"  Semantic IoU     : "
            f"{semantic_iou:.3f} >= "
            f"{semantic_threshold:.3f} -> PASS"
        )

    else:

        print(
            f"  Semantic IoU     : "
            f"{semantic_iou:.3f} < "
            f"{semantic_threshold:.3f} -> FAIL"
        )

    # FID
    if fid_result["score"] is None:

        print(
            "  FID              : "
            f"{fid_result['reason']}"
        )

    else:

        fid_score = (
            fid_result["score"]
        )

        if fid_score < fid_threshold:

            print(
                f"  FID              : "
                f"{fid_score:.3f} < "
                f"{fid_threshold:.2f} -> PASS"
            )

        else:

            print(
                f"  FID              : "
                f"{fid_score:.3f} >= "
                f"{fid_threshold:.2f} -> FAIL"
            )

    print()

    print("-" * 78)

    print(
        f"FINAL DECISION : {status}"
    )

    print("-" * 78)

    print()


# ============================================================
# JSON REPORT
# ============================================================

def build_json_report(
    original_path,
    generated_path,
    status,
    image_info,
    edge_metrics,
    bbox_result,
    semantic_iou,
    fid_result,
    edge_threshold,
    bbox_threshold,
    count_threshold,
    semantic_threshold,
    fid_threshold
):

    edge_score = (
        edge_metrics[
            "edge_recall_chance_corrected"
        ]
    )

    edge_status = evaluate_metric(
        edge_score,
        edge_threshold,
        ">="
    )

    if bbox_result is None:

        bbox_status = None
        count_status = None

    else:

        bbox_status = (
            "PASS"
            if bbox_result["hard_pass"]
            else "FAIL"
        )

        count_status = evaluate_metric(
            bbox_result["count_drift"],
            count_threshold,
            "<="
        )

    semantic_status = evaluate_metric(
        semantic_iou,
        semantic_threshold,
        ">="
    )

    fid_status = evaluate_metric(
        fid_result["score"],
        fid_threshold,
        "<"
    )

    return {

        "status": status,

        "images": {
            "original": original_path,
            "generated": generated_path
        },

        "image_info": image_info,

        "metrics": {

            "structural_edge_score": round(
                edge_score,
                4
            ),

            "edge_recall_tolerant": round(
                edge_metrics[
                    "edge_recall_tolerant"
                ],
                4
            ),

            "edge_chance_level": round(
                edge_metrics[
                    "chance_level"
                ],
                4
            ),

            "lit_fraction": round(
                edge_metrics[
                    "lit_fraction"
                ],
                4
            ),

            "bbox_iou": (
                bbox_result["mean_iou"]
                if bbox_result is not None
                else None
            ),

            "object_count_drift": (
                bbox_result[
                    "count_drift"
                ]
                if bbox_result is not None
                else None
            ),

            "semantic_iou": (
                round(
                    semantic_iou,
                    4
                )
                if semantic_iou is not None
                else None
            ),

            "fid": (
                fid_result["score"]
            )
        },

        "thresholds": {

            "structural_edge_score": (
                edge_threshold
            ),

            "per_object_iou": (
                bbox_threshold
            ),

            "object_count_drift": (
                count_threshold
            ),

            "semantic_iou": (
                semantic_threshold
                if semantic_threshold is not None
                else "TBD"
            ),

            "fid": fid_threshold
        },

        "checks": {

            "image_size": (
                "PASS"
                if not image_info["resized"]
                else "WARNING"
            ),

            "structural_edges": (
                edge_status
            ),

            "hard_per_object_bbox": (
                bbox_status
            ),

            "object_count": (
                count_status
            ),

            "semantic_iou": (
                semantic_status
            ),

            "fid": (
                fid_status
            )
        },

        "bbox_details": (
            bbox_result
            if bbox_result is not None
            else {
                "status": "NOT_EVALUATED",
                "reason": (
                    "Bounding-box annotations "
                    "were not provided."
                )
            }
        ),

        "fid_details": fid_result,

        "decision_reasons": {

            "structural_edges": (
                f"{edge_score:.4f} "
                f">= {edge_threshold:.4f}"
                if edge_status == "PASS"
                else (
                    f"{edge_score:.4f} "
                    f"< {edge_threshold:.4f}"
                )
            ),

            "bbox": (
                "Not evaluated — "
                "bounding-box annotations "
                "were not provided."
                if bbox_result is None
                else (
                    "All objects passed the "
                    "hard per-object IoU "
                    "requirement and no new "
                    "objects were detected."
                    if bbox_result["hard_pass"]
                    else (
                        "One or more objects "
                        "failed the hard IoU "
                        "requirement or new "
                        "objects were detected."
                    )
                )
            ),

            "object_count": (
                "Not evaluated — "
                "bounding-box annotations "
                "were not provided."
                if bbox_result is None
                else (
                    f"Count drift = "
                    f"{bbox_result['count_drift']}, "
                    f"allowed = "
                    f"{count_threshold}."
                )
            ),

            "semantic_iou": (
                "Not evaluated — segmentation "
                "masks were not provided."
                if semantic_iou is None
                else (
                    "Threshold is TBD; requires "
                    "empirical ablation."
                    if semantic_threshold is None
                    else (
                        f"Semantic IoU = "
                        f"{semantic_iou:.4f}, "
                        f"threshold = "
                        f"{semantic_threshold:.4f}."
                    )
                )
            ),

            "fid": (
                fid_result["reason"]
                if fid_result["score"] is None
                else (
                    f"FID = "
                    f"{fid_result['score']:.4f}, "
                    f"target < "
                    f"{fid_threshold:.2f}."
                )
            )
        }
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Layer 4 Quality Control Report"
        )
    )

    # --------------------------------------------------------
    # Images
    # --------------------------------------------------------

    parser.add_argument(
        "--original",
        required=True,
        help="Original/reference image"
    )

    parser.add_argument(
        "--generated",
        required=True,
        help="Generated/augmented image"
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    parser.add_argument(
        "--out",
        default="test_data/qc_report.json",
        help=(
            "Output report path"
        )
    )

    # --------------------------------------------------------
    # Bounding boxes
    # --------------------------------------------------------

    parser.add_argument(
        "--orig-boxes",
        default=None,
        help=(
            "Original bounding-box JSON"
        )
    )

    parser.add_argument(
        "--gen-boxes",
        default=None,
        help=(
            "Generated bounding-box JSON"
        )
    )

    # --------------------------------------------------------
    # Masks
    # --------------------------------------------------------

    parser.add_argument(
        "--orig-mask",
        default=None,
        help=(
            "Original segmentation mask"
        )
    )

    parser.add_argument(
        "--gen-mask",
        default=None,
        help=(
            "Generated segmentation mask"
        )
    )

    # --------------------------------------------------------
    # FID datasets
    # --------------------------------------------------------

    parser.add_argument(
        "--real-dir",
        default=None,
        help=(
            "Real/reference dataset folder"
        )
    )

    parser.add_argument(
        "--generated-dir",
        default=None,
        help=(
            "Generated dataset folder"
        )
    )

    # --------------------------------------------------------
    # Thresholds
    # --------------------------------------------------------

    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=DEFAULT_EDGE_THRESHOLD
    )

    parser.add_argument(
        "--bbox-threshold",
        type=float,
        default=DEFAULT_BBOX_THRESHOLD
    )

    parser.add_argument(
        "--count-drift",
        type=int,
        default=DEFAULT_COUNT_DRIFT
    )

    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=DEFAULT_SEMANTIC_THRESHOLD
    )

    parser.add_argument(
        "--fid-threshold",
        type=float,
        default=DEFAULT_FID_THRESHOLD
    )

    args = parser.parse_args()

    # ========================================================
    # LOAD IMAGES
    # ========================================================

    try:

        original = load_image(
            args.original
        )

        generated = load_image(
            args.generated
        )

    except Exception as e:

        print(
            f"[ERROR] {e}"
        )

        return 1

    # ========================================================
    # RESIZE GENERATED IMAGE
    # ========================================================

    generated_resized, image_info = (
        resize_to_reference(
            original,
            generated
        )
    )

    if image_info["resized"]:

        print(
            "[INFO] Image dimensions differ."
        )

        print(
            "[INFO] Resizing generated image "
            "to original dimensions."
        )

    # ========================================================
    # ROBUST EDGE METRIC
    # ========================================================

    edge_metrics = (
        compute_edge_metrics_robust(
            original,
            generated_resized
        )
    )

    # ========================================================
    # BBOX
    # ========================================================

    bbox_result = None

    if (
        args.orig_boxes
        and args.gen_boxes
    ):

        try:

            original_boxes = load_boxes(
                args.orig_boxes
            )

            generated_boxes = load_boxes(
                args.gen_boxes
            )

            bbox_result = (
                evaluate_bounding_boxes(
                    original_boxes,
                    generated_boxes,
                    args.bbox_threshold
                )
            )

        except Exception as e:

            print(
                "[WARNING] BBox evaluation "
                f"failed: {e}"
            )

    elif (
        args.orig_boxes
        or args.gen_boxes
    ):

        print(
            "[WARNING] Both "
            "--orig-boxes and "
            "--gen-boxes are required."
        )

    # ========================================================
    # SEMANTIC IoU
    # ========================================================

    semantic_iou = None

    if (
        args.orig_mask
        and args.gen_mask
    ):

        try:

            original_mask = load_mask(
                args.orig_mask
            )

            generated_mask = load_mask(
                args.gen_mask
            )

            semantic_iou = (
                compute_semantic_iou(
                    original_mask,
                    generated_mask
                )
            )

        except Exception as e:

            print(
                "[WARNING] Semantic IoU "
                f"failed: {e}"
            )

    # ========================================================
    # FID
    # ========================================================

    fid_result = (
        compute_fid_if_requested(
            args.real_dir,
            args.generated_dir
        )
    )

    # ========================================================
    # DETERMINE STATUS
    # ========================================================

    image_status = (
        "PASS"
        if not image_info["resized"]
        else "WARNING"
    )

    edge_score = (
        edge_metrics[
            "edge_recall_chance_corrected"
        ]
    )

    edge_status = evaluate_metric(
        edge_score,
        args.edge_threshold,
        ">="
    )

    if bbox_result is None:

        bbox_status = "N/E"
        count_status = "N/E"

    else:

        bbox_status = (
            "PASS"
            if bbox_result["hard_pass"]
            else "FAIL"
        )

        count_status = evaluate_metric(
            bbox_result["count_drift"],
            args.count_drift,
            "<="
        )

    semantic_status = evaluate_metric(
        semantic_iou,
        args.semantic_threshold,
        ">="
    )

    fid_status = evaluate_metric(
        fid_result["score"],
        args.fid_threshold,
        "<"
    )

    status = determine_final_status(
        [
            image_status,
            edge_status,
            bbox_status,
            count_status,
            semantic_status,
            fid_status
        ]
    )

    # ========================================================
    # PRINT REPORT
    # ========================================================

    print_report(
        original_path=args.original,
        generated_path=args.generated,
        status=status,
        image_info=image_info,
        edge_metrics=edge_metrics,
        bbox_result=bbox_result,
        semantic_iou=semantic_iou,
        fid_result=fid_result,
        edge_threshold=args.edge_threshold,
        bbox_threshold=args.bbox_threshold,
        count_threshold=args.count_drift,
        semantic_threshold=args.semantic_threshold,
        fid_threshold=args.fid_threshold
    )

    # ========================================================
    # JSON REPORT
    # ========================================================

    report = build_json_report(
        original_path=args.original,
        generated_path=args.generated,
        status=status,
        image_info=image_info,
        edge_metrics=edge_metrics,
        bbox_result=bbox_result,
        semantic_iou=semantic_iou,
        fid_result=fid_result,
        edge_threshold=args.edge_threshold,
        bbox_threshold=args.bbox_threshold,
        count_threshold=args.count_drift,
        semantic_threshold=args.semantic_threshold,
        fid_threshold=args.fid_threshold
    )

    # ========================================================
    # SAVE REPORT
    # ========================================================

    output_dir = os.path.dirname(
        os.path.abspath(args.out)
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    with open(
        args.out,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2
        )

    print(
        f"Report saved to: {args.out}"
    )

    return 0


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    raise SystemExit(
        main()
    )