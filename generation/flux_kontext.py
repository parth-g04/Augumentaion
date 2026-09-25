"""FLUX.1 Kontext generative backend for Layer 4.

Implements the flow-based generative augmentation backend using
FLUX.1 Kontext (Black Forest Labs), per the 16 September faculty
review pivoting Layer 4 from GAN-primary to flow-based
diffusion-primary generation.

This backend is intended to run on the DGX B200. Model loading is
lazy (happens on first call to generate()), so importing this module
on machines without a GPU will not fail.
"""

import time
from typing import Optional

import numpy as np
import torch
from PIL import Image
from diffusers import FluxKontextPipeline

from .interfaces import AugmentedOutput, BaseGenerator, ConditionConfig


class FluxKontextGenerator(BaseGenerator):
    """Flow-based generative augmentation using FLUX.1 Kontext.

    Takes an existing RGB observation plus a weather/domain instruction
    and produces an edited observation using a real flow-matching
    generative model, as opposed to the procedural baseline in
    generator_adapter.py.
    """

    def __init__(self, model_id: str = "black-forest-labs/FLUX.1-Kontext-dev"):
        self.model_id = model_id
        self.pipe = None  # Lazily loaded on first generate() call

    def _load_pipeline(self) -> None:
        """Loads the FLUX.1 Kontext pipeline via Diffusers onto GPU."""
        if self.pipe is not None:
            return
        self.pipe = FluxKontextPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
        )
        self.pipe.to("cuda")

    def _build_prompt(self, condition: ConditionConfig) -> str:
        """Builds a text editing instruction from the condition config.

        Only 'clear -> rain' and 'clear -> fog' are targeted first,
        per the phased plan. Other weather types can be extended later.
        """
        weather_prompts = {
            "rain": "make the scene rainy, wet ground, visible rain streaks in the air",
            "fog": "make the scene foggy, reduced visibility, atmospheric haze",
            "haze": "add atmospheric haze to the scene",
            "clear": "keep the scene clear with no weather change",
        }
        return weather_prompts.get(condition.weather, "keep the scene clear")

    def generate(
        self,
        base_rgb: np.ndarray,
        condition: ConditionConfig,
        depth: Optional[np.ndarray] = None,
        semantic_mask: Optional[np.ndarray] = None,
        edges: Optional[np.ndarray] = None
    ) -> AugmentedOutput:
        """Applies FLUX.1 Kontext generative augmentation to a base observation.

        depth, semantic_mask, and edges are accepted for interface
        compatibility with BaseGenerator but are not consumed by this
        backend (FLUX.1 Kontext operates on image + text instruction
        only). Structural conditioning via these inputs is handled by
        the separate ControlledDiffusionGenerator backend.
        """
        self._load_pipeline()

        torch.cuda.reset_peak_memory_stats()
        start_t = time.perf_counter()

        input_image = Image.fromarray(base_rgb)
        prompt = self._build_prompt(condition)
        generator = torch.Generator("cuda").manual_seed(condition.seed)

        result = self.pipe(
            image=input_image,
            prompt=prompt,
            generator=generator,
        )
        output_image = np.array(result.images[0])

        latency_ms = (time.perf_counter() - start_t) * 1000.0
        vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        return AugmentedOutput(
            image=output_image,
            condition=condition,
            inference_time_ms=round(latency_ms, 2),
            vram_allocated_mb=round(vram_mb, 2),
            metadata={
                "model_id": self.model_id,
                "seed": condition.seed,
                "prompt": prompt,
                "resolution": f"{output_image.shape[1]}x{output_image.shape[0]}",
                "gpu": torch.cuda.get_device_name(0),
            }
        )
