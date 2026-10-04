import sys, math, time, argparse
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/home/24jainm/person1_3dgs/renderer/3DGS_PoseRender")
import utils
import camera as camera_mod
from camera import Camera
from gaussian_model import GaussianModel
from render import Renderer

ap = argparse.ArgumentParser()
ap.add_argument("--az", type=float, default=0.0,   help="azimuth deg: 0 = camera south of scene looking north, 90 = west of scene, 180 = north, 270 = east")
ap.add_argument("--el", type=float, default=45.0,  help="elevation deg: 0 = level with scene, 89 = straight down")
ap.add_argument("--dist", type=float, default=1.0, help="distance from scene center, in units of the scene's horizontal extent (smaller = closer)")
ap.add_argument("--fov", type=float, default=75.0, help="horizontal field of view in degrees (smaller = zoomed in)")
ap.add_argument("--out", type=str, default="/home/24jainm/person1_3dgs/cuda_render_angle.png")
a = ap.parse_args()

camera_mod.get_projection_matrix = lambda fx, fy: utils.get_projection_matrix(fx, fy, znear=0.01, zfar=1000.0)

PLY = "/home/24jainm/person1_3dgs/overhead_bridge/data_3d/overhead_bridge/3dgs_ply/point_cloud_utm50.ply"

t0 = time.time()
model = GaussianModel().load(PLY)
print("loaded", model.means3D.shape[0], "gaussians in", round(time.time() - t0, 1), "s")

xyz = model.means3D.detach().cpu().numpy().astype(np.float64)
center = xyz.mean(axis=0)
extent = xyz.max(axis=0) - xyz.min(axis=0)
radius = a.dist * max(extent[0], extent[1])

az, el = math.radians(a.az), math.radians(min(a.el, 89.0))
offset = np.array([math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)])
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