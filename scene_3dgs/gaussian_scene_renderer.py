"""Wrapper and bridge for upstream 3DGS_PoseRender implementation.
Wraps the actual 3DGS_PoseRender components without reimplementing or replacing the rasterizer.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np

# Ensure third_party/3DGS_PoseRender is available in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_UPSTREAM_DIR = _PROJECT_ROOT / "third_party" / "3DGS_PoseRender"

if _UPSTREAM_DIR.exists() and str(_UPSTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM_DIR))


def check_cuda_environment() -> Tuple[bool, str]:
    """Checks whether the environment has CUDA and required rasterizer packages.
    Adheres to the non-negotiable research rules: never pretend a model ran if it didn't.
    """
    try:
        import torch
    except ImportError:
        return False, "PyTorch is not installed in the environment."

    if not torch.cuda.is_available():
        return False, "CUDA is not available. 3DGS_PoseRender requires an NVIDIA GPU with CUDA support."

    try:
        import diff_gaussian_rasterization  # noqa: F401
    except ImportError:
        return False, (
            "diff-gaussian-rasterization is not installed or compiled. "
            "Compile it from 'third_party/3DGS_PoseRender/diff-gaussian-rasterization'."
        )

    return True, "CUDA environment is ready for 3DGS_PoseRender."


class PoseRenderBackend:
    """Encapsulates the actual upstream 3DGS_PoseRender pipeline."""

    def __init__(self):
        self._cuda_available, self._cuda_message = check_cuda_environment()

    @property
    def is_cuda_available(self) -> bool:
        return self._cuda_available

    @property
    def status_message(self) -> str:
        return self._cuda_message

    def load_gaussian_model(self, ply_path: str):
        """Loads Gaussian point cloud using upstream 3DGS_PoseRender GaussianModel."""
        if not self._cuda_available:
            raise RuntimeError(
                f"Cannot load GaussianModel on this environment: {self._cuda_message}"
            )

        # Import actual upstream GaussianModel
        from gaussian_model import GaussianModel  # type: ignore

        model = GaussianModel().load(ply_path)
        return model

    def render(
        self,
        gaussian_model: Any,
        cam_info: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Invokes upstream 3DGS_PoseRender Renderer to render an RGB frame.

        Args:
            gaussian_model: Loaded upstream GaussianModel.
            cam_info: Dictionary containing width, height, fx, fy, position, rotation.

        Returns:
            Tuple of (rgb_uint8_array [H, W, 3], render_telemetry_dict)
        """
        if not self._cuda_available:
            raise RuntimeError(
                f"Cannot execute 3DGS_PoseRender rasterizer: {self._cuda_message}"
            )

        import torch
        from camera import Camera  # type: ignore
        from render import Renderer  # type: ignore

        camera = Camera()
        camera.load(cam_info)

        renderer = Renderer(gaussian_model, camera, logging=False)
        rendered_tensor = renderer.render()  # Returns [3, H, W] float32 on CUDA

        # Convert to numpy uint8 [H, W, 3]
        clamped = torch.clamp(rendered_tensor, 0.0, 1.0)
        rgb_np = clamped.detach().cpu().numpy().transpose(1, 2, 0)
        rgb_uint8 = (rgb_np * 255.0).astype(np.uint8)

        telemetry = {
            "backend": "3DGS_PoseRender",
            "upstream_image_width": camera.image_width,
            "upstream_image_height": camera.image_height,
            "tensor_shape": list(rendered_tensor.shape),
            "device": str(rendered_tensor.device)
        }

        return rgb_uint8, telemetry
