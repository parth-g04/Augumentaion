from typing import List, Dict, Any, Tuple
import numpy as np

from .interfaces import BBox2D


def compute_bbox_iou_and_drift(
    orig_boxes: List[BBox2D],
    aug_boxes: List[BBox2D],
    min_match_iou: float = 0.1
) -> Tuple[float, int]:

    count_drift = abs(
        len(orig_boxes) - len(aug_boxes)
    )

    if not orig_boxes and not aug_boxes:
        return 1.0, 0

    if not orig_boxes or not aug_boxes:
        return 0.0, count_drift

    ious = []
    unmatched_aug = list(aug_boxes)

    for b_orig in orig_boxes:

        best_iou = 0.0
        best_idx = -1

        for idx, b_aug in enumerate(unmatched_aug):

            if b_aug.class_name == b_orig.class_name:

                iou = b_orig.compute_iou(b_aug)

                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

        if best_idx >= 0 and best_iou > min_match_iou:

            ious.append(best_iou)
            unmatched_aug.pop(best_idx)

        else:

            ious.append(0.0)

    mean_iou = (
        float(np.mean(ious))
        if ious
        else 0.0
    )

    return mean_iou, count_drift


# ============================================================
# HARD PER-OBJECT IOU
# ============================================================

def compute_per_object_bbox_check(
    orig_boxes: List[BBox2D],
    aug_boxes: List[BBox2D],
    min_iou: float = 0.60,
    min_match_iou: float = 0.10
) -> Dict[str, Any]:

    """
    Hard per-object IoU check.

    Every original object must have a matching generated object
    of the same class with IoU >= min_iou.

    This is different from simply checking mean IoU.
    """

    unmatched_aug = list(aug_boxes)

    objects = []

    for index, original in enumerate(orig_boxes):

        best_iou = 0.0
        best_index = -1

        for j, generated in enumerate(unmatched_aug):

            if generated.class_name != original.class_name:
                continue

            iou = original.compute_iou(
                generated
            )

            if iou > best_iou:
                best_iou = iou
                best_index = j

        if best_index >= 0 and best_iou >= min_match_iou:

            matched = unmatched_aug.pop(
                best_index
            )

            # ------------------------------------------------
            # Object center displacement
            # ------------------------------------------------

            orig_cx = (
                original.xmin + original.xmax
            ) / 2.0

            orig_cy = (
                original.ymin + original.ymax
            ) / 2.0

            aug_cx = (
                matched.xmin + matched.xmax
            ) / 2.0

            aug_cy = (
                matched.ymin + matched.ymax
            ) / 2.0

            displacement = float(
                np.sqrt(
                    (orig_cx - aug_cx) ** 2
                    +
                    (orig_cy - aug_cy) ** 2
                )
            )

            passed = (
                best_iou >= min_iou
            )

            objects.append({
                "object_index": index + 1,
                "class_name": original.class_name,
                "iou": round(best_iou, 4),
                "center_displacement": round(
                    displacement,
                    2
                ),
                "passed": passed
            })

        else:

            objects.append({
                "object_index": index + 1,
                "class_name": original.class_name,
                "iou": 0.0,
                "center_displacement": None,
                "passed": False
            })

    # --------------------------------------------------------
    # New objects in generated image
    # --------------------------------------------------------

    new_objects = []

    for generated in unmatched_aug:

        new_objects.append({
            "class_name": generated.class_name,
            "xmin": generated.xmin,
            "ymin": generated.ymin,
            "xmax": generated.xmax,
            "ymax": generated.ymax
        })

    # --------------------------------------------------------
    # Final hard decision
    # --------------------------------------------------------

    all_objects_pass = all(
        obj["passed"]
        for obj in objects
    )

    no_new_objects = (
        len(new_objects) == 0
    )

    hard_bbox_pass = (
        all_objects_pass
        and no_new_objects
    )

    return {
        "objects": objects,

        "new_objects": new_objects,

        "matched_object_count": len(
            objects
        ),

        "new_object_count": len(
            new_objects
        ),

        "all_objects_pass": all_objects_pass,

        "no_new_objects": no_new_objects,

        "hard_bbox_pass": hard_bbox_pass,

        "threshold": min_iou
    }