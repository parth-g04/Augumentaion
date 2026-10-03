import sys, json, argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
from generation.flux_kontext import FluxKontextGenerator
from generation.interfaces import ConditionConfig

PRESETS = {
    "rain": "make the scene rainy, wet ground, visible rain streaks in the air",
    "fog": "make the scene foggy, reduced visibility, atmospheric haze",
    "night": "make the scene at night, dark sky, artificial lighting visible",
    "golden_hour": ("make the scene during golden hour, warm soft directional sunlight, long shadows, "
                    "warm amber and orange tones, no visible sun disc, no lens flare, "
                    "realistic aerial drone photograph"),
}

ap = argparse.ArgumentParser()
ap.add_argument("--inp", required=True)
ap.add_argument("--condition", choices=list(PRESETS), help="preset prompt")
ap.add_argument("--prompt", help="custom prompt (overrides --condition)")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--guidance", type=float, default=2.5)
ap.add_argument("--out", required=True)
a = ap.parse_args()
a.out = a.out.rstrip("~")

prompt = a.prompt or PRESETS.get(a.condition)
if not prompt:
    sys.exit("give --condition or --prompt")

src = Image.open(a.inp).convert("RGB")
w0, h0 = src.size
s = 1024 / max(w0, h0)
w, h = int(round(w0 * s / 16)) * 16, int(round(h0 * s / 16)) * 16
base_rgb = np.array(src.resize((w, h), Image.LANCZOS))

gen = FluxKontextGenerator()
condition = ConditionConfig(seed=a.seed)

out = gen.generate(base_rgb, condition, prompt_override=prompt, guidance_scale=a.guidance)

Image.fromarray(out.image).save(a.out)

meta = dict(out.metadata)
meta["input_image"] = a.inp
meta["input_size"] = [w0, h0]
meta["output_size"] = [w, h]
meta["inference_time_ms"] = out.inference_time_ms
meta["vram_allocated_mb"] = out.vram_allocated_mb
meta["torch"] = torch.__version__

json.dump(meta, open(a.out + ".json", "w"), indent=2)
print(meta)