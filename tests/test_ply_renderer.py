"""Unit tests for PlySceneRenderer and upstream 3DGS_PoseRender adapter."""

from pathlib import Path
import numpy as np
import plyfile
import pytest

from scene_3dgs.interfaces import CameraIntrinsics, CameraPose, RenderOutput, SceneRenderer
from scene_3dgs.ply_renderer import PlySceneRenderer
from scene_3dgs.gaussian_scene_renderer import PoseRenderBackend, check_cuda_environment


def create_minimal_ply(file_path: Path):
    """Helper to create a valid minimal 3DGS PLY point cloud for testing geometry parsing."""
    num_verts = 3
    # Property values for 3 Gaussians
    x = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    y = np.array([100.0, 200.0, 300.0], dtype=np.float32)
    z = np.array([-5.0, 0.0, 5.0], dtype=np.float32)
    nx = np.zeros(num_verts, dtype=np.float32)
    ny = np.zeros(num_verts, dtype=np.float32)
    nz = np.zeros(num_verts, dtype=np.float32)
    f_dc_0 = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    f_dc_1 = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    f_dc_2 = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    opacity = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    scale_0 = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    scale_1 = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    scale_2 = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    rot_0 = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    rot_1 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    rot_2 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    rot_3 = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    vertex_data = np.empty(
        num_verts,
        dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
            ('opacity', 'f4'),
            ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
            ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
        ]
    )
    vertex_data['x'] = x
    vertex_data['y'] = y
    vertex_data['z'] = z
    vertex_data['nx'] = nx
    vertex_data['ny'] = ny
    vertex_data['nz'] = nz
    vertex_data['f_dc_0'] = f_dc_0
    vertex_data['f_dc_1'] = f_dc_1
    vertex_data['f_dc_2'] = f_dc_2
    vertex_data['opacity'] = opacity
    vertex_data['scale_0'] = scale_0
    vertex_data['scale_1'] = scale_1
    vertex_data['scale_2'] = scale_2
    vertex_data['rot_0'] = rot_0
    vertex_data['rot_1'] = rot_1
    vertex_data['rot_2'] = rot_2
    vertex_data['rot_3'] = rot_3

    el = plyfile.PlyElement.describe(vertex_data, 'vertex')
    plyfile.PlyData([el]).write(str(file_path))


def test_ply_renderer_file_not_found():
    with pytest.raises(FileNotFoundError):
        PlySceneRenderer(ply_path="non_existent_scene.ply")


def test_ply_renderer_geometry_parsing(tmp_path):
    ply_file = tmp_path / "test_points.ply"
    create_minimal_ply(ply_file)

    renderer = PlySceneRenderer(ply_path=str(ply_file), scene_id="unit_test_scene")

    assert isinstance(renderer, SceneRenderer)
    assert renderer.num_gaussians == 3

    # Bounds must match actual data
    bounds = renderer.get_scene_bounds()
    assert bounds["x"] == (10.0, 30.0)
    assert bounds["y"] == (100.0, 300.0)
    assert bounds["z"] == (-5.0, 5.0)

    # Centroid must match mean of data (never assumed to be at origin)
    centroid = renderer.get_scene_centroid()
    assert pytest.approx(centroid[0], 1e-4) == 20.0
    assert pytest.approx(centroid[1], 1e-4) == 200.0
    assert pytest.approx(centroid[2], 1e-4) == 0.0


def test_ply_renderer_cuda_error_handling(tmp_path):
    ply_file = tmp_path / "test_points.ply"
    create_minimal_ply(ply_file)

    renderer = PlySceneRenderer(ply_path=str(ply_file))
    pose = CameraPose(x=20.0, y=150.0, z=20.0, pitch=45.0, yaw=0.0)

    # When CUDA is unavailable, the adapter must raise an error and NEVER emit fake images
    if not renderer.backend.is_cuda_available:
        with pytest.raises(RuntimeError) as exc_info:
            renderer.render(pose)
        assert "Real 3DGS rendering requires an NVIDIA GPU" in str(exc_info.value)


def test_ply_renderer_metadata_and_provenance(tmp_path, monkeypatch):
    ply_file = tmp_path / "test_points.ply"
    create_minimal_ply(ply_file)

    renderer = PlySceneRenderer(ply_path=str(ply_file), scene_id="meta_test_scene")

    # Simulate available CUDA and mock upstream renderer execution
    monkeypatch.setattr(renderer.backend, "_cuda_available", True)
    monkeypatch.setattr(
        renderer.backend,
        "load_gaussian_model",
        lambda p: "mock_model_handle"
    )

    # Mock 3DGS_PoseRender producing downscaled 160x120 output from requested 640x480
    mock_rendered_rgb = np.zeros((120, 160, 3), dtype=np.uint8)
    monkeypatch.setattr(
        renderer.backend,
        "render",
        lambda m, c: (mock_rendered_rgb, {"device": "cuda:0", "backend": "3DGS_PoseRender"})
    )

    pose = CameraPose(x=10.0, y=20.0, z=30.0, pitch=60.0, yaw=90.0)
    intrinsics = CameraIntrinsics(width=640, height=480, fov_deg=75.0)

    output = renderer.render(pose, intrinsics)

    assert isinstance(output, RenderOutput)
    assert output.rgb.shape == (120, 160, 3)

    # Verify research integrity rules:
    # 1. Annotations must be empty (Person 1 does not invent bounding boxes)
    assert output.annotations == []
    # 2. Semantic mask is None (no fake masks)
    assert output.semantic_mask is None
    # 3. Depth source is none
    assert output.depth_source == "none"

    # Verify provenance metadata
    meta = output.metadata
    assert meta["scene_source"] == "public_3dgs"
    assert meta["scene_id"] == "meta_test_scene"
    assert meta["num_gaussians"] == 3
    assert meta["requested_resolution"] == {"width": 640, "height": 480}
    assert meta["actual_output_resolution"] == {"width": 160, "height": 120}
    assert meta["camera_pose"]["x"] == 10.0
    assert meta["camera_pose"]["pitch"] == 60.0
