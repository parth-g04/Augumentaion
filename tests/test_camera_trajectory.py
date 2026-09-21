"""Unit tests for camera trajectory generator."""

import pytest
from scene_3dgs.trajectory import TrajectoryConfig, TrajectoryGenerator


def test_trajectory_orbit_generation():
    config = TrajectoryConfig(altitude_min=40.0, altitude_max=120.0, radius=50.0)
    gen = TrajectoryGenerator(config)

    poses = gen.generate_orbit(num_views=5, altitude=80.0, pitch=45.0)
    assert len(poses) == 5

    # Verify altitudes and pitches match
    for p in poses:
        assert p.z == 80.0
        assert p.pitch == 45.0
        # Radius check: sqrt(x^2 + y^2) == 50.0
        r_dist = (p.x**2 + p.y**2)**0.5
        assert abs(r_dist - 50.0) < 1e-4


def test_altitude_and_pitch_sweeps():
    config = TrajectoryConfig(altitude_min=50.0, altitude_max=100.0, pitch_min=30.0, pitch_max=60.0)
    gen = TrajectoryGenerator(config)

    alt_poses = gen.generate_altitude_sweep(num_steps=3)
    assert len(alt_poses) == 3
    assert alt_poses[0].z == 50.0
    assert alt_poses[1].z == 75.0
    assert alt_poses[2].z == 100.0

    pitch_poses = gen.generate_pitch_sweep(num_steps=4)
    assert len(pitch_poses) == 4
    assert pitch_poses[0].pitch == 30.0
    assert pitch_poses[-1].pitch == 60.0


def test_linear_flight_path():
    gen = TrajectoryGenerator()
    path = gen.generate_linear_flight_path(
        start_pt=(-100.0, 0.0),
        end_pt=(100.0, 0.0),
        num_views=5,
        altitude=90.0
    )
    assert len(path) == 5
    assert path[0].x == -100.0
    assert path[-1].x == 100.0
    for p in path:
        assert p.z == 90.0
