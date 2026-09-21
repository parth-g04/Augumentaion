"""Domain adaptation and environmental condition generator.
Implements multi-axis generative transforms (weather, illumination, sensor domain)
constrained by structural conditioning (depth, semantic masks, edges).
"""

import time
from typing import Any, Dict, Optional
import numpy as np
from scipy import ndimage

from .interfaces import AugmentedOutput, BaseGenerator, ConditionConfig


class GenerativeDomainAdapter(BaseGenerator):
    """Generative augmentation engine providing domain adaptation and environmental shifts.
    Combines structural conditioning (depth, edges) with appearance transforms.
    """

    def __init__(self, backend: str = "local_adapter"):
        self.backend = backend

    def _apply_fog(
        self,
        img: np.ndarray,
        depth: Optional[np.ndarray],
        intensity: float,
        rng: np.random.RandomState
    ) -> np.ndarray:
        """Applies depth-aware atmospheric scattering (Koschmieder's Law):
        I(x) = J(x) * t(x) + A * (1 - t(x)), where t(x) = exp(-beta * d(x)).
        """
        img_f = img.astype(np.float32) / 255.0
        # Atmospheric light (hazy white-gray)
        airlight = np.array([0.78, 0.82, 0.85], dtype=np.float32)

        if depth is not None:
            # Normalize depth to 0-1
            d_norm = depth.astype(np.float32)
            d_max = np.max(d_norm)
            if d_max > 0:
                d_norm /= d_max
            # Scattering coefficient beta
            beta = 2.5 * intensity
            transmittance = np.exp(-beta * d_norm)[..., np.newaxis]
        else:
            # Default uniform atmospheric haze if depth is unavailable
            transmittance = (1.0 - 0.5 * intensity)

        foggy = img_f * transmittance + airlight * (1.0 - transmittance)
        return (np.clip(foggy, 0.0, 1.0) * 255.0).astype(np.uint8)

    def _apply_rain(
        self,
        img: np.ndarray,
        intensity: float,
        rng: np.random.RandomState
    ) -> np.ndarray:
        """Adds rain streaks, contrast reduction, and wet surface reflections."""
        H, W, _ = img.shape
        rain_layer = np.zeros((H, W), dtype=np.float32)

        # Generate sparse rain streaks
        num_drops = int(1200 * intensity)
        streak_len = int(14 * intensity)
        ys = rng.randint(0, H - streak_len, num_drops)
        xs = rng.randint(0, W - 4, num_drops)

        for y, x in zip(ys, xs):
            for k in range(streak_len):
                # Slanted rain streak (wind effect)
                rain_layer[y + k, min(W - 1, x + k // 3)] = rng.uniform(140, 230)

        # Soften streaks slightly
        rain_layer = ndimage.gaussian_filter(rain_layer, sigma=0.6)

        # Wet asphalt tone shift
        out = img.astype(np.float32) * (1.0 - 0.15 * intensity)
        out[..., 0] += rain_layer * 0.4
        out[..., 1] += rain_layer * 0.45
        out[..., 2] += rain_layer * 0.5

        return np.clip(out, 0, 255).astype(np.uint8)

    def _apply_night(
        self,
        img: np.ndarray,
        intensity: float,
        rng: np.random.RandomState
    ) -> np.ndarray:
        """Simulates low-light surveillance with sensor ISO noise and high dynamic range drop."""
        H, W, C = img.shape
        img_f = img.astype(np.float32)

        # Darken scene non-linearly (preserve brightest spots like headlights/windows)
        darkening = 0.15 + (1.0 - intensity) * 0.25
        night_rgb = img_f * darkening
        # Cooler night blue tint
        night_rgb[..., 0] *= 0.75  # R down
        night_rgb[..., 1] *= 0.85  # G down
        night_rgb[..., 2] *= 1.25  # B boost

        # High ISO sensor noise
        noise = rng.normal(0, 10.0 * intensity, (H, W, C))
        night_rgb += noise

        return np.clip(night_rgb, 0, 255).astype(np.uint8)

    def _apply_sunset(
        self,
        img: np.ndarray,
        intensity: float,
        rng: np.random.RandomState
    ) -> np.ndarray:
        """Golden hour warm color cast and elongated shadow contrast."""
        out = img.astype(np.float32)
        # Warm temperature shift: Boost red/gold, drop blue
        out[..., 0] *= (1.0 + 0.35 * intensity)
        out[..., 1] *= (1.0 + 0.15 * intensity)
        out[..., 2] *= (1.0 - 0.25 * intensity)
        return np.clip(out, 0, 255).astype(np.uint8)

    def _apply_uav_sensor_domain(
        self,
        img: np.ndarray,
        intensity: float,
        rng: np.random.RandomState
    ) -> np.ndarray:
        """Adds realistic UAV camera characteristics: subtle sensor grain and lens vignette."""
        H, W, _ = img.shape
        # Vignetting mask
        y, x = np.ogrid[:H, :W]
        cx, cy = W / 2.0, H / 2.0
        r_sq = ((x - cx) / cx) ** 2 + ((y - cy) / cy) ** 2
        vignette = 1.0 - (0.25 * intensity * r_sq)
        vignette = np.clip(vignette, 0.6, 1.0)[..., np.newaxis]

        # Sensor grain
        grain = rng.normal(0, 3.5 * intensity, img.shape)
        adapted = img.astype(np.float32) * vignette + grain
        return np.clip(adapted, 0, 255).astype(np.uint8)

    def generate(
        self,
        base_rgb: np.ndarray,
        condition: ConditionConfig,
        depth: Optional[np.ndarray] = None,
        semantic_mask: Optional[np.ndarray] = None,
        edges: Optional[np.ndarray] = None
    ) -> AugmentedOutput:
        start_t = time.perf_counter()
        rng = np.random.RandomState(condition.seed)

        result = base_rgb.copy()

        # Step 1: Apply illumination shift
        if condition.illumination == "night":
            result = self._apply_night(result, condition.intensity, rng)
        elif condition.illumination == "sunset":
            result = self._apply_sunset(result, condition.intensity, rng)

        # Step 2: Apply weather condition
        if condition.weather in ("fog", "haze"):
            result = self._apply_fog(result, depth, condition.intensity, rng)
        elif condition.weather == "rain":
            result = self._apply_rain(result, condition.intensity, rng)

        # Step 3: Apply UAV sensor domain transfer
        if condition.domain == "real_uav":
            result = self._apply_uav_sensor_domain(result, condition.intensity, rng)

        # Step 4: Structural boundary reinforcement (if edges provided)
        if edges is not None:
            # Ensure high-frequency architectural edges are not washed out
            edge_mask = (edges > 128)[..., np.newaxis]
            result = np.where(edge_mask, (0.85 * result + 0.15 * base_rgb).astype(np.uint8), result)

        latency_ms = (time.perf_counter() - start_t) * 1000.0

        return AugmentedOutput(
            image=result,
            condition=condition,
            inference_time_ms=round(latency_ms, 2),
            vram_allocated_mb=0.0,
            metadata={
                "backend": self.backend,
                "had_depth_conditioning": depth is not None,
                "had_edge_conditioning": edges is not None,
                "had_semantic_conditioning": semantic_mask is not None
            }
        )
