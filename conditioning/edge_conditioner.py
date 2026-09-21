"""Edge and boundary conditioning module for Layer 4.
Extracts high-frequency structural boundaries to constrain building geometry,
road edges, and vehicular silhouettes.
"""

from typing import Tuple
import numpy as np
from scipy import ndimage


class EdgeConditioner:
    """Computes structural edge maps from RGB views and depth maps."""

    def __init__(self, low_threshold: float = 30.0, high_threshold: float = 100.0):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def extract_sobel_edges(self, image_rgb: np.ndarray) -> np.ndarray:
        """Extracts gradient edges from an RGB image using Sobel filters.
        Returns uint8 edge map in [0, 255].
        """
        # Convert to grayscale luminance
        if image_rgb.ndim == 3:
            gray = 0.299 * image_rgb[..., 0] + 0.587 * image_rgb[..., 1] + 0.114 * image_rgb[..., 2]
        else:
            gray = image_rgb.astype(np.float32)

        gx = ndimage.sobel(gray, axis=1)
        gy = ndimage.sobel(gray, axis=0)
        grad = np.hypot(gx, gy)

        # Normalize and threshold
        max_grad = np.max(grad)
        if max_grad > 1e-6:
            grad = (grad / max_grad) * 255.0

        edges = (grad > self.low_threshold).astype(np.uint8) * 255
        return edges

    def combine_rgb_and_depth_edges(
        self,
        rgb_edges: np.ndarray,
        depth_edges: np.ndarray
    ) -> np.ndarray:
        """Merges optical boundary edges with geometric depth discontinuity edges."""
        combined = np.maximum(rgb_edges, depth_edges)
        return combined
