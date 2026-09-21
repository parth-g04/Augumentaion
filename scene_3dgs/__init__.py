"""3D Gaussian Splatting and camera rendering module for Layer 4."""

from .interfaces import (
    CameraPose,
    CameraIntrinsics,
    RenderOutput,
    SceneRenderer
)
from .trajectory import TrajectoryConfig, TrajectoryGenerator
from .surrogate_renderer import SurrogateSceneRenderer

__all__ = [
    "CameraPose",
    "CameraIntrinsics",
    "RenderOutput",
    "SceneRenderer",
    "TrajectoryConfig",
    "TrajectoryGenerator",
    "SurrogateSceneRenderer"
]
