"""Core interfaces for 3D Gaussian Splatting and surrogate scene renderers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple
import math
import numpy as np


@dataclass(frozen=True)
class CameraPose:
    """6-DOF camera pose in world coordinates."""
    x: float
    y: float
    z: float  # Altitude above ground level (m)
    roll: float = 0.0  # degrees
    pitch: float = 45.0  # degrees (30° oblique to 85° nadir)
    yaw: float = 0.0  # degrees (0° to 360° heading)

    def to_translation_vector(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    def to_rotation_matrix(self) -> np.ndarray:
        """Computes 3x3 camera rotation matrix in world ENU coordinates.
        Columns of R correspond to [camera_right, camera_down, camera_forward].
        - Heading yaw=0 points North (+Y), 90 points East (+X).
        - Pitch tilts the optical axis from horizontal towards the ground (-Z).
        - Roll performs in-plane camera rotation.
        """
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)
        roll_rad = math.radians(self.roll)

        # Forward optical axis (Z_cam) in world coordinates
        fx = math.sin(yaw_rad) * math.cos(pitch_rad)
        fy = math.cos(yaw_rad) * math.cos(pitch_rad)
        fz = -math.sin(pitch_rad)
        uz = np.array([fx, fy, fz], dtype=np.float64)
        uz /= np.linalg.norm(uz)

        # Base camera right (X_cam) in world horizontal plane
        rx = math.cos(yaw_rad)
        ry = -math.sin(yaw_rad)
        rz = 0.0
        ux = np.array([rx, ry, rz], dtype=np.float64)
        ux /= np.linalg.norm(ux)

        # Camera down (Y_cam) in world coordinates
        uy = np.cross(uz, ux)
        uy /= np.linalg.norm(uy)

        # Apply roll rotation around optical axis (uz)
        if abs(roll_rad) > 1e-6:
            cos_r = math.cos(roll_rad)
            sin_r = math.sin(roll_rad)
            ux_rot = cos_r * ux + sin_r * uy
            uy_rot = -sin_r * ux + cos_r * uy
            ux = ux_rot / np.linalg.norm(ux_rot)
            uy = uy_rot / np.linalg.norm(uy_rot)

        return np.column_stack([ux, uy, uz])

    def to_extrinsic_matrix(self) -> np.ndarray:
        """Returns 4x4 camera extrinsics matrix [R | t]."""
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.to_rotation_matrix()
        T[:3, 3] = self.to_translation_vector()
        return T

    def to_dict(self) -> Dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "roll": self.roll,
            "pitch": self.pitch,
            "yaw": self.yaw
        }


@dataclass(frozen=True)
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    width: int = 1280
    height: int = 720
    fov_deg: float = 75.0
    fx: Optional[float] = None
    fy: Optional[float] = None
    cx: Optional[float] = None
    cy: Optional[float] = None

    def __post_init__(self):
        # Auto-compute focal length from FOV if not provided
        if self.fx is None or self.fy is None:
            fov_rad = math.radians(self.fov_deg)
            f = (self.width / 2.0) / math.tan(fov_rad / 2.0)
            object.__setattr__(self, "fx", f)
            object.__setattr__(self, "fy", f)
        if self.cx is None:
            object.__setattr__(self, "cx", self.width / 2.0)
        if self.cy is None:
            object.__setattr__(self, "cy", self.height / 2.0)

    def to_intrinsic_matrix(self) -> np.ndarray:
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "fov_deg": self.fov_deg,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy
        }


@dataclass
class RenderOutput:
    """Standardized output from any scene renderer (surrogate or true 3DGS)."""
    rgb: np.ndarray  # HxW x 3 uint8 (0-255)
    depth: np.ndarray  # HxW float32
    depth_type: Literal["metric", "relative"]
    depth_source: str  # e.g., "3DGS_splat", "synthetic_ground_truth", "monocular"
    camera_pose: CameraPose
    camera_intrinsics: CameraIntrinsics
    semantic_mask: Optional[np.ndarray] = None  # HxW uint8
    instance_mask: Optional[np.ndarray] = None  # HxW int32
    annotations: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SceneRenderer(ABC):
    """Abstract decoupled interface for scene rendering.
    Layer 4 calls this interface without caring whether the underlying source is
    a surrogate dataset (FlyAwareV2), 3DGS_PoseRender, or the real Layer 1 output.
    """

    @abstractmethod
    def render(
        self,
        camera_pose: CameraPose,
        camera_intrinsics: Optional[CameraIntrinsics] = None
    ) -> RenderOutput:
        """Render a single viewpoint from the scene given camera pose and intrinsics."""
        pass

    @abstractmethod
    def get_scene_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Return the spatial bounding box (x, y, z) of the scene."""
        pass
