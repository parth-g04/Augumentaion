"""Unit tests for generative augmentation engine."""

import numpy as np
import pytest
from generation import ConditionConfig, GenerativeDomainAdapter


def test_generator_deterministic_seeding():
    adapter = GenerativeDomainAdapter()
    base_img = np.full((100, 100, 3), 120, dtype=np.uint8)
    cond = ConditionConfig(weather="rain", seed=42)

    out1 = adapter.generate(base_img, cond)
    out2 = adapter.generate(base_img, cond)

    assert np.array_equal(out1.image, out2.image)
    assert out1.inference_time_ms > 0


def test_depth_attenuated_fog():
    adapter = GenerativeDomainAdapter()
    base_img = np.full((100, 100, 3), 100, dtype=np.uint8)

    # Gradient depth: near on left (x=0), far on right (x=99)
    depth = np.linspace(5.0, 150.0, 100).reshape(1, 100).repeat(100, axis=0)

    cond = ConditionConfig(weather="fog", intensity=1.0, seed=123)
    out = adapter.generate(base_img, cond, depth=depth)

    # In fog, distant objects have higher atmospheric airlight transmittance loss
    # (whiter/hazier) than nearby objects
    near_mean = np.mean(out.image[:, :10])
    far_mean = np.mean(out.image[:, -10:])
    assert far_mean > near_mean


def test_night_and_sunset_illumination():
    adapter = GenerativeDomainAdapter()
    base_img = np.full((100, 100, 3), 150, dtype=np.uint8)

    night_cond = ConditionConfig(illumination="night", seed=99)
    out_night = adapter.generate(base_img, night_cond)
    # Night should be significantly darker
    assert np.mean(out_night.image) < np.mean(base_img) * 0.6

    sunset_cond = ConditionConfig(illumination="sunset", seed=99)
    out_sunset = adapter.generate(base_img, sunset_cond)
    # Sunset should boost Red channel over Blue channel
    assert np.mean(out_sunset.image[..., 0]) > np.mean(out_sunset.image[..., 2])
