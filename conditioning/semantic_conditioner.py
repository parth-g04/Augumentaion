"""Semantic conditioning module for Layer 4.
Processes multi-class segmentation masks, generates standard color palettes,
and extracts per-class spatial bounding regions.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


class SemanticConditioner:
    """Manages semantic masks and extracts structural class bounding regions."""

    # Standard palette mapping class_id -> (R, G, B, class_name)
    DEFAULT_PALETTE: Dict[int, Tuple[int, int, int, str]] = {
        0: (0, 0, 0, "background"),
        1: (70, 70, 70, "building"),
        2: (220, 20, 60, "vehicle"),
        3: (255, 0, 0, "person"),
        4: (128, 64, 128, "road"),
        5: (107, 142, 35, "vegetation")
    }

    def __init__(self, palette: Optional[Dict[int, Tuple[int, int, int, str]]] = None):
        self.palette = palette or self.DEFAULT_PALETTE

    def colorize_mask(self, semantic_mask: np.ndarray) -> np.ndarray:
        """Converts an HxW integer class mask into an HxWx3 RGB color map."""
        H, W = semantic_mask.shape
        rgb = np.zeros((H, W, 3), dtype=np.uint8)

        for class_id, (r, g, b, _) in self.palette.items():
            mask = (semantic_mask == class_id)
            if np.any(mask):
                rgb[mask] = [r, g, b]

        return rgb

    def extract_class_masks(self, semantic_mask: np.ndarray) -> Dict[str, np.ndarray]:
        """Returns a dict of binary masks (0 or 255) keyed by class name."""
        class_masks = {}
        for class_id, (_, _, _, name) in self.palette.items():
            mask = (semantic_mask == class_id).astype(np.uint8) * 255
            if np.any(mask):
                class_masks[name] = mask
        return class_masks

    def get_class_bounding_extents(
        self,
        semantic_mask: np.ndarray
    ) -> Dict[str, List[Tuple[int, int, int, int]]]:
        """Calculates bounding extents (xmin, ymin, xmax, ymax) for each class."""
        extents = {}
        for class_id, (_, _, _, name) in self.palette.items():
            ys, xs = np.where(semantic_mask == class_id)
            if len(xs) > 0:
                xmin, xmax = int(np.min(xs)), int(np.max(xs))
                ymin, ymax = int(np.min(ys)), int(np.max(ys))
                extents[name] = [(xmin, ymin, xmax, ymax)]
        return extents
