"""Generative augmentation interfaces and data schemas."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional
import numpy as np


WeatherType = Literal["clear", "rain", "fog", "haze"]
IlluminationType = Literal["day", "sunset", "night"]
DomainType = Literal["synthetic", "real_uav", "sensor_noise"]


@dataclass
class ConditionConfig:
    """Configures multi-axis environmental, lighting, and sensor domain shifts."""
    weather: WeatherType = "clear"
    illumination: IlluminationType = "day"
    domain: DomainType = "real_uav"
    intensity: float = 1.0  # 0.0 to 1.0
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weather": self.weather,
            "illumination": self.illumination,
            "domain": self.domain,
            "intensity": self.intensity,
            "seed": self.seed
        }


@dataclass
class AugmentedOutput:
    """Output from the generative augmentation engine."""
    image: np.ndarray  # HxWx3 uint8
    condition: ConditionConfig
    inference_time_ms: float
    vram_allocated_mb: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseGenerator(ABC):
    """Abstract interface for all generative augmentation backends."""

    @abstractmethod
    def generate(
        self,
        base_rgb: np.ndarray,
        condition: ConditionConfig,
        depth: Optional[np.ndarray] = None,
        semantic_mask: Optional[np.ndarray] = None,
        edges: Optional[np.ndarray] = None
    ) -> AugmentedOutput:
        """Applies generative augmentation to a base rendered observation."""
        pass
