"""Decoupled Surrogate Scene Renderer for Layer 4.
Provides realistic RGB, metric depth, semantic masks, and projected 2D annotations
without blocking on Layer 1-3 completion.
"""

from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np
from PIL import Image, ImageDraw

from .interfaces import (
    CameraIntrinsics,
    CameraPose,
    RenderOutput,
    SceneRenderer,
)


class SurrogateSceneRenderer(SceneRenderer):
    """Surrogate scene renderer that projects a 3D aerial scene onto camera views.
    Implements true 3D-to-2D perspective projection for boxes, depth calculation,
    and semantic mask rendering.
    """

    def __init__(
        self,
        scene_name: str = "surrogate_aerial_campus",
        scene_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        depth_type: str = "metric",
        depth_source: str = "surrogate_ground_truth"
    ):
        self.scene_name = scene_name
        self.depth_type = depth_type
        self.depth_source = depth_source
        self.scene_bounds = scene_bounds or {
            "x": (-150.0, 150.0),
            "y": (-150.0, 150.0),
            "z": (0.0, 50.0)
        }

        # Initialize fixed 3D objects in the scene (buildings, parked vehicles, humans)
        self._init_scene_objects()

    def _init_scene_objects(self):
        """Sets up 3D objects with center (x,y,z), dimensions (dx, dy, dz), and semantic class."""
        self.objects = [
            # Buildings (class_id 1)
            {"id": "bldg_01", "class": "building", "class_id": 1, "center": [-25.0, 30.0, 10.0], "size": [20.0, 25.0, 20.0], "color": (120, 125, 130)},
            {"id": "bldg_02", "class": "building", "class_id": 1, "center": [35.0, -20.0, 8.0], "size": [22.0, 18.0, 16.0], "color": (140, 135, 125)},
            {"id": "bldg_03", "class": "building", "class_id": 1, "center": [40.0, 45.0, 12.0], "size": [26.0, 20.0, 24.0], "color": (110, 115, 125)},
            # Vehicles (class_id 2)
            {"id": "car_01", "class": "car", "class_id": 2, "center": [5.0, 10.0, 0.75], "size": [4.5, 2.0, 1.5], "color": (180, 40, 40)},
            {"id": "car_02", "class": "car", "class_id": 2, "center": [-10.0, -5.0, 0.75], "size": [4.8, 2.1, 1.5], "color": (40, 70, 180)},
            {"id": "truck_01", "class": "truck", "class_id": 2, "center": [15.0, -15.0, 1.5], "size": [8.0, 2.5, 3.0], "color": (210, 180, 50)},
            # Pedestrians (class_id 3)
            {"id": "person_01", "class": "person", "class_id": 3, "center": [2.0, 5.0, 0.9], "size": [0.6, 0.6, 1.8], "color": (30, 170, 50)},
            {"id": "person_02", "class": "person", "class_id": 3, "center": [-2.0, 8.0, 0.9], "size": [0.6, 0.6, 1.8], "color": (160, 50, 160)},
        ]

    def get_scene_bounds(self) -> Dict[str, Tuple[float, float]]:
        return self.scene_bounds

    def project_point_to_camera(
        self,
        pt_world: np.ndarray,
        pose: CameraPose,
        intrinsics: CameraIntrinsics
    ) -> Optional[Tuple[float, float, float]]:
        """Projects a 3D world point into camera pixel coordinates (u, v) and camera depth (z_cam).
        Returns None if point is behind camera.
        """
        # World to Camera transformation:
        # Camera extrinsics: X_cam = R_cam^T * (X_world - C)
        C = pose.to_translation_vector()
        R = pose.to_rotation_matrix()
        pt_rel = pt_world - C
        pt_cam = R.T @ pt_rel

        x_c, y_c, z_c = pt_cam[0], pt_cam[1], pt_cam[2]
        if z_c <= 0.1:  # Behind camera or near clipping plane
            return None

        # Camera to image plane (Pinhole model)
        u = intrinsics.fx * (x_c / z_c) + intrinsics.cx
        v = intrinsics.fy * (y_c / z_c) + intrinsics.cy

        return (float(u), float(v), float(z_c))

    def render(
        self,
        camera_pose: CameraPose,
        camera_intrinsics: Optional[CameraIntrinsics] = None
    ) -> RenderOutput:
        """Renders RGB view, depth map, semantic mask, and 2D bounding boxes."""
        intr = camera_intrinsics or CameraIntrinsics()
        W, H = intr.width, intr.height

        # Base image: Ground terrain (asphalt + grass)
        rgb_img = Image.new("RGB", (W, H), color=(70, 95, 65))
        draw_rgb = ImageDraw.Draw(rgb_img)

        # Semantic mask: 0=background/terrain, 1=building, 2=vehicle, 3=person
        semantic_mask = np.zeros((H, W), dtype=np.uint8)
        # Depth map: initialize to far clip (e.g. 300m)
        depth_map = np.full((H, W), 300.0, dtype=np.float32)

        # Draw road intersection on ground
        road_pts_world = [
            np.array([-100.0, -10.0, 0.0]),
            np.array([100.0, -10.0, 0.0]),
            np.array([100.0, 15.0, 0.0]),
            np.array([-100.0, 15.0, 0.0])
        ]
        road_proj = [self.project_point_to_camera(p, camera_pose, intr) for p in road_pts_world]
        if all(p is not None for p in road_proj):
            poly = [(p[0], p[1]) for p in road_proj]
            draw_rgb.polygon(poly, fill=(50, 52, 55))

        # Sort objects by distance from camera for painter's rendering
        C = camera_pose.to_translation_vector()
        sorted_objs = sorted(
            self.objects,
            key=lambda o: -np.linalg.norm(np.array(o["center"]) - C)
        )

        annotations = []

        for obj in sorted_objs:
            cx, cy, cz = obj["center"]
            dx, dy, dz = obj["size"]

            # Compute 8 corners of 3D bounding box
            corners_world = []
            for sx in [-0.5, 0.5]:
                for sy in [-0.5, 0.5]:
                    for sz in [-0.5, 0.5]:
                        corners_world.append(np.array([
                            cx + sx * dx,
                            cy + sy * dy,
                            cz + sz * dz
                        ]))

            projections = [self.project_point_to_camera(p, camera_pose, intr) for p in corners_world]
            valid_proj = [p for p in projections if p is not None]

            if len(valid_proj) < 4:
                continue

            us = [p[0] for p in valid_proj]
            vs = [p[1] for p in valid_proj]
            zs = [p[2] for p in valid_proj]

            u_min, u_max = max(0, int(min(us))), min(W - 1, int(max(us)))
            v_min, v_max = max(0, int(min(vs))), min(H - 1, int(max(vs)))

            if u_max > u_min and v_max > v_min:
                mean_depth = float(np.mean(zs))
                # Draw box in RGB
                draw_rgb.rectangle([u_min, v_min, u_max, v_max], fill=obj["color"], outline=(20, 20, 20))

                # Update semantic mask & depth map
                semantic_mask[v_min:v_max + 1, u_min:u_max + 1] = obj["class_id"]
                depth_mask = depth_map[v_min:v_max + 1, u_min:u_max + 1] > mean_depth
                depth_map[v_min:v_max + 1, u_min:u_max + 1][depth_mask] = mean_depth

                annotations.append({
                    "id": obj["id"],
                    "class": obj["class"],
                    "class_id": obj["class_id"],
                    "bbox_xyxy": [u_min, v_min, u_max, v_max],
                    "bbox_xywh": [u_min, v_min, u_max - u_min, v_max - v_min],
                    "depth_m": mean_depth,
                    "center_3d": obj["center"]
                })

        rgb_array = np.array(rgb_img, dtype=np.uint8)

        return RenderOutput(
            rgb=rgb_array,
            depth=depth_map,
            depth_type=self.depth_type,
            depth_source=self.depth_source,
            camera_pose=camera_pose,
            camera_intrinsics=intr,
            semantic_mask=semantic_mask,
            annotations=annotations,
            metadata={
                "scene_name": self.scene_name,
                "num_objects_visible": len(annotations),
                "depth_type": self.depth_type,
                "depth_source": self.depth_source
            }
        )
