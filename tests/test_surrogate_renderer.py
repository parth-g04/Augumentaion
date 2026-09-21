"""Unit tests for SurrogateSceneRenderer."""

import numpy as np
import pytest
from scene_3dgs.interfaces import CameraIntrinsics, CameraPose
from scene_3dgs.surrogate_renderer import SurrogateSceneRenderer


def test_surrogate_renderer_projection():
    renderer = SurrogateSceneRenderer()
    intrinsics = CameraIntrinsics(width=640, height=480, fov_deg=75.0)

    # Place drone at (0, -80, 80) looking toward center with 45° pitch
    pose = CameraPose(x=0.0, y=-80.0, z=80.0, roll=0.0, pitch=45.0, yaw=0.0)

    output = renderer.render(pose, intrinsics)

    assert output.rgb.shape == (480, 640, 3)
    assert output.rgb.dtype == np.uint8
    assert output.depth.shape == (480, 640)
    assert output.depth.dtype == np.float32
    assert output.depth_type == "metric"
    assert output.depth_source == "surrogate_ground_truth"
    assert output.semantic_mask.shape == (480, 640)

    # Check annotations are extracted
    assert len(output.annotations) > 0
    first_ann = output.annotations[0]
    assert "class" in first_ann
    assert "bbox_xyxy" in first_ann
    assert "depth_m" in first_ann
    assert first_ann["depth_m"] > 0
