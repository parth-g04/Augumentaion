import numpy as np


def compute_mask_iou(
    orig_mask: np.ndarray,
    aug_mask: np.ndarray,
) -> float:

    if orig_mask.shape != aug_mask.shape:
        raise ValueError(
            f"Mask shape mismatch: "
            f"{orig_mask.shape} vs "
            f"{aug_mask.shape}"
        )

    original = orig_mask > 0
    generated = aug_mask > 0

    intersection = np.logical_and(
        original,
        generated
    ).sum()

    union = np.logical_or(
        original,
        generated
    ).sum()

    if union == 0:
        return 1.0

    return float(intersection / union)