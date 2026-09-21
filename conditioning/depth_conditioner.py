"""Depth conditioning module for Layer 4.
Handles metric and relative depth normalization, depth discontinuity edge extraction,
and metadata provenance tracking.
"""

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple
import numpy as np
from scipy import ndimage


@dataclass
class DepthProvenance:
    """Explicit provenance metadata for depth maps."""
    depth_type: Literal["metric", "relative"]
    source: str  # e.g., "3DGS_splat", "surrogate_ground_truth", "monocular_depth"
    min_depth: float
    max_depth: float
    is_calibrated: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "depth_type": self.depth_type,
            "source": self.source,
            "min_depth": self.min_depth,
            "max_depth": self.max_depth,
            "is_calibrated": self.is_calibrated
        }


class DepthConditioner:
    """Processes and conditions depth maps for generative domain adaptation models."""

    def __init__(self, near_clip_m: float = 1.0, far_clip_m: float = 200.0):
        self.near_clip = near_clip_m
        self.far_clip = far_clip_m

    def process(
        self,
        depth_raw: np.ndarray,
        depth_type: Literal["metric", "relative"] = "metric",
        source: str = "surrogate_ground_truth"
    ) -> Tuple[np.ndarray, DepthProvenance]:
        """Normalizes depth map to [0, 1] range and records exact provenance.
        
        For metric depth, values are clipped between near_clip and far_clip,
        then mapped to [0, 1] where 0 is closest and 1 is farthest.
        """
        valid_depth = depth_raw[np.isfinite(depth_raw)]
        min_val = float(np.min(valid_depth)) if len(valid_depth) > 0 else 0.0
        max_val = float(np.max(valid_depth)) if len(valid_depth) > 0 else 1.0

        if depth_type == "metric":
            clipped = np.clip(depth_raw, self.near_clip, self.far_clip)
            norm_depth = (clipped - self.near_clip) / (self.far_clip - self.near_clip)
            is_calib = True
        else:
            # Relative depth: normalize against min/max observed in frame
            span = max_val - min_val
            if span > 1e-6:
                norm_depth = (depth_raw - min_val) / span
            else:
                norm_depth = np.zeros_like(depth_raw)
            is_calib = False

        provenance = DepthProvenance(
            depth_type=depth_type,
            source=source,
            min_depth=min_val,
            max_depth=max_val,
            is_calibrated=is_calib
        )

        return norm_depth.astype(np.float32), provenance

    def to_uint8_map(self, norm_depth: np.ndarray) -> np.ndarray:
        """Converts normalized [0, 1] depth to 8-bit [0, 255] grayscale image."""
        return (np.clip(norm_depth, 0.0, 1.0) * 255.0).astype(np.uint8)

    def extract_depth_discontinuities(
        self,
        norm_depth: np.ndarray,
        gradient_threshold: float = 0.05
    ) -> np.ndarray:
        """Extracts sharp depth discontinuities (e.g. building borders, occluding edges)
        using Sobel filter gradients. Returns a binary mask (uint8: 0 or 255).
        """
        gx = ndimage.sobel(norm_depth, axis=1)
        gy = ndimage.sobel(norm_depth, axis=0)
        grad_mag = np.hypot(gx, gy)

        edges = (grad_mag > gradient_threshold).astype(np.uint8) * 255
        return edges
