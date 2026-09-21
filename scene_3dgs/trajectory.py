"""Camera trajectory generator for 6-DOF drone viewpoints.
Generates systematic sweeps (orbital, altitude, pitch, linear flight path)
for producing diverse camera observations from a single 3D scene.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import numpy as np

from .interfaces import CameraPose


@dataclass
class TrajectoryConfig:
    """Configurable experimental ranges for drone camera sweeps.
    These are experiment parameters, not fixed project-wide assumptions.
    """
    altitude_min: float = 40.0
    altitude_max: float = 120.0
    pitch_min: float = 30.0   # Oblique
    pitch_max: float = 85.0   # Near-nadir
    yaw_min: float = 0.0
    yaw_max: float = 360.0
    center_x: float = 0.0
    center_y: float = 0.0
    radius: float = 80.0      # Orbit radius in meters


class TrajectoryGenerator:
    """Generates 6-DOF camera poses based on user/experiment configurations."""

    def __init__(self, config: Optional[TrajectoryConfig] = None):
        self.config = config or TrajectoryConfig()

    def generate_orbit(
        self,
        num_views: int = 5,
        altitude: float = 80.0,
        pitch: float = 45.0,
        radius: Optional[float] = None
    ) -> List[CameraPose]:
        """Generates an orbital sweep around the scene center at a fixed altitude and pitch.
        
        Args:
            num_views: Number of viewpoints to sample evenly around the 360° orbit.
            altitude: Drone flight altitude (m).
            pitch: Camera pitch down angle in degrees.
            radius: Orbit radius from center in meters.
        """
        r = radius if radius is not None else self.config.radius
        poses = []
        angles = np.linspace(0, 360, num_views, endpoint=False)

        for angle in angles:
            rad = math.radians(angle)
            x = self.config.center_x + r * math.sin(rad)
            y = self.config.center_y - r * math.cos(rad)
            # Camera points towards center (center_x, center_y)
            dx = self.config.center_x - x
            dy = self.config.center_y - y
            cam_yaw = math.degrees(math.atan2(dx, dy)) % 360.0

            poses.append(CameraPose(
                x=float(x),
                y=float(y),
                z=float(altitude),
                roll=0.0,
                pitch=float(pitch),
                yaw=float(cam_yaw)
            ))
        return poses

    def generate_altitude_sweep(
        self,
        num_steps: int = 5,
        x: float = 0.0,
        y: float = -60.0,
        pitch: float = 45.0,
        yaw: float = 0.0
    ) -> List[CameraPose]:
        """Sweeps drone altitude from altitude_min to altitude_max."""
        altitudes = np.linspace(self.config.altitude_min, self.config.altitude_max, num_steps)
        return [
            CameraPose(x=x, y=y, z=float(alt), roll=0.0, pitch=pitch, yaw=yaw)
            for alt in altitudes
        ]

    def generate_pitch_sweep(
        self,
        num_steps: int = 5,
        x: float = 0.0,
        y: float = -60.0,
        altitude: float = 80.0,
        yaw: float = 0.0
    ) -> List[CameraPose]:
        """Sweeps camera pitch from oblique (pitch_min) to nadir (pitch_max)."""
        pitches = np.linspace(self.config.pitch_min, self.config.pitch_max, num_steps)
        return [
            CameraPose(x=x, y=y, z=altitude, roll=0.0, pitch=float(p), yaw=yaw)
            for p in pitches
        ]

    def generate_linear_flight_path(
        self,
        start_pt: Tuple[float, float],
        end_pt: Tuple[float, float],
        num_views: int = 5,
        altitude: float = 80.0,
        pitch: float = 45.0,
        yaw: Optional[float] = None
    ) -> List[CameraPose]:
        """Simulates a straight-line drone flight across the survey zone."""
        xs = np.linspace(start_pt[0], end_pt[0], num_views)
        ys = np.linspace(start_pt[1], end_pt[1], num_views)

        # Compute heading from path direction if yaw is not specified
        if yaw is None:
            dx = end_pt[0] - start_pt[0]
            dy = end_pt[1] - start_pt[1]
            path_yaw = math.degrees(math.atan2(dy, dx)) % 360.0
        else:
            path_yaw = yaw

        return [
            CameraPose(x=float(x), y=float(y), z=altitude, roll=0.0, pitch=pitch, yaw=float(path_yaw))
            for x, y in zip(xs, ys)
        ]
