"""Camera-angle-controlled renderer for real 3DGS .ply scenes.

Uses the upstream 3DGS_PoseRender rasterizer (GaussianModel, Camera, Renderer)
directly. This is the exact approach validated on the DGX B200 against the
overhead_bridge scene: it produced the first real, recognizable render after
the CUDA toolkit and diff-gaussian-rasterization were built successfully.

Requires:
  - A CUDA GPU with diff-gaussian-rasterization compiled (see README for setup).
  - The upstream 3DGS_PoseRender repo available on sys.path (set via
    --renderer-dir, defaulting to third_party/3DGS_PoseRender relative to
    this repo's root).

Camera is placed using azimuth/elevation/distance relative to the scene's
own bounding-box center and extent, NOT absolute world coordinates — this
is the placement strategy that was actually validated, since real 3DGS
scenes can have wildly different coordinate ranges.
"""
import sys
import math
import time
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True, help="path to the 3DGS .ply file")
ap.add_argument("--renderer-dir", default=None,
                 help="path to upstream 3DGS_PoseRender repo "
                      "(default: third_party/3DGS_PoseRender next to this repo root)")
ap.add_argument("--az", type=float, default=0.0,
                 help="azimuth deg: 0 = camera south of scene looking north, "
                      "90 = west of scene, 180 = north, 270 = east")
ap.add_argument("--el", type=float, default=45.0,
                 help="elevation deg: 0 = level with scene, 89 = straight down")
ap.add_argument("--dist", type=float, default=1.0,
                 help="distance from scene center, in units of the scene's "
                      "horizontal extent (smaller = closer)")
ap.add_argument("--fov", type=float, default=75.0,
                 help="horizontal field of view in degrees (smaller = zoomed in)")
ap.add_argument("--zfar", type=float, default=1000.0,
                 help="far clipping plane; upstream default of 100 clips/blacks "
                      "out large real-world scenes")
ap.add_argument("--out", required=True)
a = ap.parse_args()

repo_root = Path(__file__).resolve().parent.parent
renderer_dir = Path(a.renderer_dir) if a.renderer_dir else repo_root / "third_party" / "3DGS_PoseRender"
if not renderer_dir.exists():
    sys.exit(
        f"Upstream renderer not found at {renderer_dir}. "
        "Clone and build 3DGS_PoseRender there first (see README), "
        "or pass --renderer-dir to point at an existing build."
    )
sys.path.insert(0, str(renderer_dir))

import utils
import camera as camera_mod
from camera import Camera
from gaussian_model import GaussianModel
from render import Renderer

# Upstream default far plane is 100 units, which clips/blacks out large
# real-world scenes (overhead_bridge is ~350 units wide). Validated override: 1000.
camera_mod.get_projection_matrix = lambda fx, fy: utils.get_projection_matrix(
    fx, fy, znear=0.01, zfar=a.zfar
)

t0 = time.time()
model = GaussianModel().load(a.ply)
print("loaded", model.means3D.shape[0], "gaussians in", round(time.time() - t0, 1), "s")

xyz = model.means3D.detach().cpu().numpy().astype(np.float64)
center = xyz.mean(axis=0)
extent = xyz.max(axis=0) - xyz.min(axis=0)
radius = a.dist * max(extent[0], extent[1])

az, el = math.radians(a.az), math.radians(min(a.el, 89.0))
offset = np.array([
    math.sin(az) * math.cos(el),
    -math.cos(az) * math.cos(el),
    math.sin(el),
])
pos = center + radius * offset

fwd = center - pos
fwd /= np.linalg.norm(fwd)
world_up = np.array([0.0, 0.0, 1.0])
right = np.cross(fwd, world_up)
right /= np.linalg.norm(right)
up = np.cross(right, fwd)
R = np.stack([right, -up, fwd], axis=1)  # columns: right, down, forward

W, H = 6400, 4800  # renderer outputs ~1/4 of this: 1600 x 1200
fx = W / (2 * math.tan(math.radians(a.fov) / 2))
cam = Camera().load({"position": pos, "rotation": R, "fx": fx, "fy": fx, "width": W, "height": H})

renderer = Renderer(model, cam, logging=False)
with torch.no_grad():
    img = renderer.render()

arr = (img.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
print("camera pos", pos.round(1), "| az", a.az, "el", a.el, "dist", a.dist, "fov", a.fov)
print("image shape", arr.shape, "non-black fraction", round(float((arr.sum(axis=2) > 0).mean()), 3))
Image.fromarray(arr).save(a.out)
print("saved", a.out)