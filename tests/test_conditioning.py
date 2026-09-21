"""Unit tests for Phase 2 structural conditioning engine."""

import numpy as np
import pytest
from conditioning import DepthConditioner, DepthProvenance, SemanticConditioner, EdgeConditioner


def test_depth_conditioner_metric_and_relative():
    conditioner = DepthConditioner(near_clip_m=5.0, far_clip_m=105.0)

    # Synthetic metric depth gradient
    raw_depth = np.linspace(5.0, 105.0, 100).reshape(10, 10).astype(np.float32)
    norm_depth, prov = conditioner.process(raw_depth, depth_type="metric", source="test_metric")

    assert norm_depth.shape == (10, 10)
    assert np.isclose(norm_depth[0, 0], 0.0)
    assert np.isclose(norm_depth[-1, -1], 1.0)
    assert prov.depth_type == "metric"
    assert prov.source == "test_metric"
    assert prov.is_calibrated is True

    # Relative depth
    rel_raw = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    norm_rel, prov_rel = conditioner.process(rel_raw, depth_type="relative", source="monocular_depth")
    assert np.isclose(norm_rel[0, 0], 0.0)
    assert np.isclose(norm_rel[-1, -1], 1.0)
    assert prov_rel.depth_type == "relative"
    assert prov_rel.is_calibrated is False


def test_depth_discontinuities():
    conditioner = DepthConditioner()
    # Step depth map (e.g. wall in front of background)
    depth = np.zeros((50, 50), dtype=np.float32)
    depth[:, 25:] = 1.0  # Sharp step at x=25

    edges = conditioner.extract_depth_discontinuities(depth, gradient_threshold=0.1)
    assert edges.shape == (50, 50)
    # The discontinuity should show up around column 25
    assert np.any(edges[:, 23:27] == 255)
    # Uniform regions should have no edges
    assert np.all(edges[:, :20] == 0)


def test_semantic_conditioner():
    conditioner = SemanticConditioner()
    mask = np.zeros((60, 60), dtype=np.uint8)
    mask[10:30, 10:30] = 1  # building
    mask[40:50, 40:50] = 2  # vehicle

    rgb_map = conditioner.colorize_mask(mask)
    assert rgb_map.shape == (60, 60, 3)
    assert rgb_map.dtype == np.uint8
    # Building pixel color check (70, 70, 70)
    assert np.all(rgb_map[20, 20] == [70, 70, 70])
    # Vehicle pixel color check (220, 20, 60)
    assert np.all(rgb_map[45, 45] == [220, 20, 60])

    class_masks = conditioner.extract_class_masks(mask)
    assert "building" in class_masks
    assert "vehicle" in class_masks
    assert np.sum(class_masks["building"] == 255) == 20 * 20

    extents = conditioner.get_class_bounding_extents(mask)
    assert "building" in extents
    assert extents["building"][0] == (10, 10, 29, 29)


def test_edge_conditioner():
    edge_cond = EdgeConditioner(low_threshold=20.0)
    # Image with sharp contrast square
    img = np.zeros((60, 60, 3), dtype=np.uint8)
    img[15:45, 15:45] = 255

    edges = edge_cond.extract_sobel_edges(img)
    assert edges.shape == (60, 60)
    assert edges.dtype == np.uint8
    assert np.any(edges > 0)

    # Combined edges
    depth_edges = np.zeros((60, 60), dtype=np.uint8)
    depth_edges[10, :] = 255
    combined = edge_cond.combine_rgb_and_depth_edges(edges, depth_edges)
    assert np.all(combined[10, :] == 255)
