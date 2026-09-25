"""Render a single viewpoint from an actual 3DGS .ply scene.
Person 1 milestone:
actual .ply -> 3DGS_PoseRender -> one explicit camera pose -> actual RGB render + metadata.
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is in python path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from PIL import Image
from scene_3dgs.interfaces import CameraIntrinsics, CameraPose
from scene_3dgs.ply_renderer import PlySceneRenderer


def render_single_view(
    ply_path: str,
    x: float,
    y: float,
    z: float,
    yaw: float,
    pitch: float,
    roll: float = 0.0,
    width: int = 1280,
    height: int = 720,
    fov_deg: float = 75.0,
    scene_id: str = None,
    output_dir: str = "data/renders/single_view"
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Person 1] Initializing 3DGS .ply renderer for: {ply_path}")
    renderer = PlySceneRenderer(ply_path=ply_path, scene_id=scene_id)

    # Explicit camera pose provided by user/caller (no invented trajectory)
    pose = CameraPose(
        x=float(x),
        y=float(y),
        z=float(z),
        roll=float(roll),
        pitch=float(pitch),
        yaw=float(yaw)
    )

    intrinsics = CameraIntrinsics(
        width=int(width),
        height=int(height),
        fov_deg=float(fov_deg)
    )

    print(f"[Person 1] Rendering viewpoint from camera position ({x}, {y}, {z}), pitch={pitch}°, yaw={yaw}°...")
    render_output = renderer.render(camera_pose=pose, camera_intrinsics=intrinsics)

    # Save output RGB image
    rgb_file = out_dir / "render_rgb.png"
    Image.fromarray(render_output.rgb).save(rgb_file)
    print(f"[Person 1] Saved rendered RGB image: {rgb_file.as_posix()}")

    # Save metadata JSON
    meta_file = out_dir / "render_metadata.json"
    metadata_payload = {
        "scene_id": render_output.metadata.get("scene_id"),
        "scene_source": render_output.metadata.get("scene_source"),
        "ply_path": render_output.metadata.get("ply_path"),
        "num_gaussians": render_output.metadata.get("num_gaussians"),
        "scene_bounds": render_output.metadata.get("scene_bounds"),
        "scene_centroid": render_output.metadata.get("scene_centroid"),
        "camera_pose": render_output.camera_pose.to_dict(),
        "camera_intrinsics": render_output.camera_intrinsics.to_dict(),
        "requested_resolution": render_output.metadata.get("requested_resolution"),
        "actual_output_resolution": render_output.metadata.get("actual_output_resolution"),
        "rgb_path": str(rgb_file.as_posix()),
        "annotations": render_output.annotations,
        "render_duration_s": render_output.metadata.get("render_duration_s"),
        "telemetry": render_output.metadata.get("telemetry")
    }

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=2)

    print(f"[Person 1] Saved render metadata: {meta_file.as_posix()}")
    return metadata_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render a single viewpoint from a 3DGS .ply scene via 3DGS_PoseRender"
    )
    parser.add_argument("--ply", required=True, help="Path to 3DGS .ply point cloud")
    parser.add_argument("--scene-id", default=None, help="Identifier for scene (defaults to ply filename)")
    parser.add_argument("--x", type=float, required=True, help="Camera X coordinate in world space")
    parser.add_argument("--y", type=float, required=True, help="Camera Y coordinate in world space")
    parser.add_argument("--z", type=float, required=True, help="Camera Z coordinate in world space")
    parser.add_argument("--yaw", type=float, default=0.0, help="Camera yaw heading in degrees")
    parser.add_argument("--pitch", type=float, default=45.0, help="Camera pitch down in degrees")
    parser.add_argument("--roll", type=float, default=0.0, help="Camera roll in degrees")
    parser.add_argument("--width", type=int, default=1280, help="Requested image width")
    parser.add_argument("--height", type=int, default=720, help="Requested image height")
    parser.add_argument("--fov", type=float, default=75.0, help="Camera field of view in degrees")
    parser.add_argument("--output-dir", default="data/renders/single_view", help="Directory to save renders")

    args = parser.parse_args()

    render_single_view(
        ply_path=args.ply,
        x=args.x,
        y=args.y,
        z=args.z,
        yaw=args.yaw,
        pitch=args.pitch,
        roll=args.roll,
        width=args.width,
        height=args.height,
        fov_deg=args.fov,
        scene_id=args.scene_id,
        output_dir=args.output_dir
    )
