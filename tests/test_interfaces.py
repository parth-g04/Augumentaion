"""Unit tests for core 3dgs interfaces and data contracts."""

import numpy as np
import pytest
from scene_3dgs.interfaces import CameraPose, CameraIntrinsics, RenderOutput


def test_camera_pose_matrices():
    pose = CameraPose(x=10.0, y=20.0, z=80.0, roll=0.0, pitch=45.0, yaw=90.0)
    
    t_vec = pose.to_translation_vector()
    assert t_vec.shape == (3,)
    assert np.allclose(t_vec, [10.0, 20.0, 80.0])

    R_mat = pose.to_rotation_matrix()
    assert R_mat.shape == (3, 3)
    # Check orthogonality of rotation matrix: R * R.T == I, det(R) == 1
    assert np.allclose(R_mat @ R_mat.T, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(R_mat), 1.0, atol=1e-6)

    ext = pose.to_extrinsic_matrix()
    assert ext.shape == (4, 4)
    assert np.allclose(ext[:3, 3], [10.0, 20.0, 80.0])


def test_camera_intrinsics():
    intr = CameraIntrinsics(width=1920, height=1080, fov_deg=75.0)
    assert intr.width == 1920
    assert intr.height == 1080
    assert intr.cx == 960.0
    assert intr.cy == 540.0
    assert intr.fx > 0 and intr.fy > 0

    K = intr.to_intrinsic_matrix()
    assert K.shape == (3, 3)
    assert K[0, 0] == intr.fx
    assert K[1, 1] == intr.fy
    assert K[0, 2] == 960.0
    assert K[1, 2] == 540.0


def test_render_output_structure():
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = np.ones((480, 640), dtype=np.float32) * 50.0
    pose = CameraPose(x=0, y=0, z=100)
    intr = CameraIntrinsics()

    out = RenderOutput(
        rgb=rgb,
        depth=depth,
        depth_type="metric",
        depth_source="surrogate_ground_truth",
        camera_pose=pose,
        camera_intrinsics=intr,
        metadata={"test": 123}
    )

    assert out.rgb.shape == (480, 640, 3)
    assert out.depth.shape == (480, 640)
    assert out.depth_type == "metric"
    assert out.depth_source == "surrogate_ground_truth"
    assert out.metadata["test"] == 123
