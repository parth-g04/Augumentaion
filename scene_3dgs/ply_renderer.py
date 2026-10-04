"""Adapter for real 3DGS .ply scene rendering via 3DGS_PoseRender.
Adheres strictly to the SceneRenderer interface and the non-negotiable research rules.
"""

from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple
import numpy as np
import plyfile

from .interfaces import (
    CameraIntrinsics,
    CameraPose,
    RenderOutput,
    SceneRenderer,
)
from .gaussian_scene_renderer import PoseRenderBackend, check_cuda_environment


class PlySceneRenderer(SceneRenderer):
    """Adapter that accepts an actual 3DGS .ply point cloud and renders viewpoints
    using the upstream 3DGS_PoseRender rasterizer.
    """

    def __init__(
        self,
        ply_path: str,
        scene_id: Optional[str] = None
    ):
        self.ply_path = Path(ply_path).resolve()
        if not self.ply_path.exists():
            raise FileNotFoundError(f"3DGS .ply file not found: {self.ply_path}")

        self.scene_id = scene_id or self.ply_path.stem
        self.backend = PoseRenderBackend()
        self._gaussian_model = None

        # Inspect and extract true geometric bounds and centroid from the .ply
        # We compute this directly from the Gaussian data rather than assuming any coordinate center.
        self._parse_ply_geometry()

    def _parse_ply_geometry(self):
        """Reads Gaussian vertex properties to determine true count, bounds, and centroid."""
        plydata = plyfile.PlyData.read(str(self.ply_path))
        vertex_element = plydata.elements[0]

        xs = np.asarray(vertex_element["x"], dtype=np.float32)
        ys = np.asarray(vertex_element["y"], dtype=np.float32)
        zs = np.asarray(vertex_element["z"], dtype=np.float32)

        self.num_gaussians = int(len(xs))
        self.scene_bounds = {
            "x": (float(np.min(xs)), float(np.max(xs))),
            "y": (float(np.min(ys)), float(np.max(ys))),
            "z": (float(np.min(zs)), float(np.max(zs)))
        }
        self.scene_centroid = (
            float(np.mean(xs)),
            float(np.mean(ys)),
            float(np.mean(zs))
        )

    def get_scene_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Return the spatial bounding box (x, y, z) of the 3DGS scene."""
        return self.scene_bounds

    def get_scene_centroid(self) -> Tuple[float, float, float]:
        """Return the calculated 3D centroid of the scene Gaussians.
        Note: The centroid is recorded for metadata reference and must NOT be used
        to automatically invent a camera trajectory.
        """
        return self.scene_centroid

    def render(
        self,
        camera_pose: CameraPose,
        camera_intrinsics: Optional[CameraIntrinsics] = None
    ) -> RenderOutput:
        """Renders an actual RGB observation from the supplied 6-DOF camera pose.

        Args:
            camera_pose: Explicit 6-DOF camera pose (position and orientation).
            camera_intrinsics: Camera intrinsic parameters (resolution, focal length, FOV).

        Returns:
            RenderOutput with real RGB render and provenance metadata.

        Raises:
            RuntimeError: If CUDA or diff-gaussian-rasterization is not available.
                         Never produces fake images or mocks.
        """
        if not self.backend.is_cuda_available:
            raise RuntimeError(
                f"3DGS rendering failed: {self.backend.status_message} "
                "Real 3DGS rendering requires an NVIDIA GPU with CUDA 11.8 and diff-gaussian-rasterization. "
                "No fake images are generated in accordance with research integrity rules."
            )

        intr = camera_intrinsics or CameraIntrinsics()

        # Build cam_info for 3DGS_PoseRender camera model
        position = [float(camera_pose.x), float(camera_pose.y), float(camera_pose.z)]
        rotation = camera_pose.to_rotation_matrix().tolist()

        cam_info = {
            "width": intr.width,
            "height": intr.height,
            "fx": intr.fx,
            "fy": intr.fy,
            "position": position,
            "rotation": rotation
        }

        # Lazy model loading into GPU
        if self._gaussian_model is None:
            self._gaussian_model = self.backend.load_gaussian_model(str(self.ply_path))

        start_time = time.perf_counter()
        rgb_array, telemetry = self.backend.render(self._gaussian_model, cam_info)
        render_duration_s = time.perf_counter() - start_time

        # 3DGS_PoseRender does not modify upstream downscaling behavior.
        # We record both requested and actual output resolutions transparently.
        actual_h, actual_w, _ = rgb_array.shape

        # Upstream 3DGS_PoseRender renders RGB only. No depth or annotations are invented.
        metadata = {
            "scene_source": "public_3dgs",
            "scene_id": self.scene_id,
            "ply_path": str(self.ply_path),
            "num_gaussians": self.num_gaussians,
            "scene_bounds": self.scene_bounds,
            "scene_centroid": {
                "x": self.scene_centroid[0],
                "y": self.scene_centroid[1],
                "z": self.scene_centroid[2]
            },
            "camera_pose": camera_pose.to_dict(),
            "camera_intrinsics": intr.to_dict(),
            "requested_resolution": {
                "width": intr.width,
                "height": intr.height
            },
            "actual_output_resolution": {
                "width": actual_w,
                "height": actual_h
            },
            "render_duration_s": round(render_duration_s, 4),
            "telemetry": telemetry
        }

        # Empty depth placeholder marked clearly as 'none' provenance
        dummy_depth = np.zeros((actual_h, actual_w), dtype=np.float32)

        return RenderOutput(
            rgb=rgb_array,
            depth=dummy_depth,
            depth_type="relative",
            depth_source="none",
            camera_pose=camera_pose,
            camera_intrinsics=intr,
            semantic_mask=None,
            instance_mask=None,
            annotations=[],  # No fake bounding boxes
            metadata=metadata
        )
