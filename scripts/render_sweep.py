"""Script to render a controlled camera sweep from the surrogate scene.
Saves RGB frames, depth maps, semantic masks, and camera metadata JSON.
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure prototype root is in python path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
from PIL import Image

from scene_3dgs.interfaces import CameraIntrinsics
from scene_3dgs.trajectory import TrajectoryConfig, TrajectoryGenerator
from scene_3dgs.surrogate_renderer import SurrogateSceneRenderer


def run_sweep(
    output_dir: str = "data/renders",
    num_views: int = 5,
    altitude: float = 80.0,
    pitch: float = 45.0,
    radius: float = 75.0,
    width: int = 640,
    height: int = 480
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    config = TrajectoryConfig(radius=radius)
    generator = TrajectoryGenerator(config)
    renderer = SurrogateSceneRenderer()
    intrinsics = CameraIntrinsics(width=width, height=height, fov_deg=75.0)

    poses = generator.generate_orbit(
        num_views=num_views,
        altitude=altitude,
        pitch=pitch,
        radius=radius
    )

    metadata_list = []

    print(f"Rendering {num_views} camera sweep views...")
    for idx, pose in enumerate(poses, start=1):
        render_output = renderer.render(pose, intrinsics)

        view_stem = f"view_{idx:02d}"
        rgb_file = out_path / f"{view_stem}.png"
        depth_npy = out_path / f"{view_stem}_depth.npy"
        depth_png = out_path / f"{view_stem}_depth.png"
        semantic_png = out_path / f"{view_stem}_semantic.png"

        # Save RGB image
        Image.fromarray(render_output.rgb).save(rgb_file)

        # Save metric depth as numpy float32 and normalized 8-bit visualization
        np.save(depth_npy, render_output.depth)
        # Normalize depth for visual check (clip 0-150m)
        d_norm = np.clip(render_output.depth / 150.0 * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(d_norm).save(depth_png)

        # Save semantic mask visualization
        if render_output.semantic_mask is not None:
            # Map semantic classes to distinct grayscale levels for visualization
            s_vis = (render_output.semantic_mask * 60).astype(np.uint8)
            Image.fromarray(s_vis).save(semantic_png)

        view_meta = {
            "view_id": view_stem,
            "rgb_path": str(rgb_file.as_posix()),
            "depth_path": str(depth_npy.as_posix()),
            "semantic_path": str(semantic_png.as_posix()),
            "camera_pose": pose.to_dict(),
            "camera_intrinsics": intrinsics.to_dict(),
            "depth_provenance": {
                "depth_type": render_output.depth_type,
                "source": render_output.depth_source,
                "min_depth_m": float(np.min(render_output.depth)),
                "max_depth_m": float(np.max(render_output.depth))
            },
            "annotations": render_output.annotations
        }
        metadata_list.append(view_meta)
        print(f"  Saved {view_stem}: {len(render_output.annotations)} objects visible.")

    meta_file = out_path / "camera_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump({
            "scene_name": renderer.scene_name,
            "num_views": num_views,
            "views": metadata_list
        }, f, indent=2)

    print(f"Complete. Metadata saved to {meta_file.as_posix()}")
    return metadata_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render camera sweep views from surrogate scene")
    parser.add_argument("--output-dir", default="data/renders", help="Directory to save rendered views")
    parser.add_argument("--num-views", type=int, default=5, help="Number of views in orbit")
    parser.add_argument("--altitude", type=float, default=80.0, help="Altitude in meters")
    parser.add_argument("--pitch", type=float, default=45.0, help="Camera pitch angle in degrees")
    parser.add_argument("--radius", type=float, default=75.0, help="Orbit radius in meters")
    args = parser.parse_args()

    run_sweep(
        output_dir=args.output_dir,
        num_views=args.num_views,
        altitude=args.altitude,
        pitch=args.pitch,
        radius=args.radius
    )
