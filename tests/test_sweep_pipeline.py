"""Integration test for camera sweep and metadata generation."""

import json
from pathlib import Path
import pytest
from scripts.render_sweep import run_sweep


def test_render_sweep_pipeline(tmp_path):
    out_dir = tmp_path / "sweep_out"
    metadata = run_sweep(
        output_dir=str(out_dir),
        num_views=3,
        altitude=60.0,
        pitch=50.0,
        radius=50.0,
        width=320,
        height=240
    )

    assert len(metadata) == 3
    assert (out_dir / "camera_metadata.json").exists()

    with open(out_dir / "camera_metadata.json", "r", encoding="utf-8") as f:
        meta_json = json.load(f)

    assert meta_json["num_views"] == 3
    assert len(meta_json["views"]) == 3

    for idx in range(1, 4):
        stem = f"view_{idx:02d}"
        assert (out_dir / f"{stem}.png").exists()
        assert (out_dir / f"{stem}_depth.npy").exists()
        assert (out_dir / f"{stem}_depth.png").exists()
        assert (out_dir / f"{stem}_semantic.png").exists()
