import sys, json, time, argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
from generation.flux_kontext import FluxKontextGenerator

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
inp = src.resize((w, h), Image.LANCZOS)

gen = FluxKontextGenerator()
gen._load_pipeline()
g = torch.Generator(device="cuda").manual_seed(a.seed)

t0 = time.time()
res = gen.pipe(image=inp, prompt=prompt, height=h, width=w, guidance_scale=a.guidance, generator=g)
ms = (time.time() - t0) * 1000
out = res.images[0]
out.save(a.out)

meta = {
    "model_id": gen.model_id, "seed": a.seed, "prompt": prompt, "guidance_scale": a.guidance,
    "input_image": a.inp, "input_size": [w0, h0], "output_size": list(out.size),
    "inference_time_ms": round(ms, 1),
    "vram_allocated_mb": round(torch.cuda.memory_allocated() / 2**20, 1),
    "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
}
json.dump(meta, open(a.out + ".json", "w"), indent=2)
print(meta)