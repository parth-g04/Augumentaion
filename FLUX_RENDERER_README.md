PERSON 2 - FLUX GENERATION AND 3DGS CAMERA RENDERING



This document covers the work done for Layer 4 generative augmentation

(FLUX.1 Kontext weather/lighting edits) and the camera-angle rendering

scripts used to produce real UAV-style input images, as tested and

confirmed on the DGX B200.



FILES IN THIS FOLDER



FLUX generation (scripts folder):



flux_edit.py

General-purpose FLUX weather/lighting edit script. Takes an input

image and either a preset condition (rain, fog, night, golden_hour)

or a custom prompt argument, runs it through FluxKontextGenerator,

and saves the output image plus a json metadata sidecar (model ID,

seed, prompt, guidance scale, input and output sizes, inference time,

VRAM, GPU).



run_batch.sh

Batch script that runs all 4 city_best images through 5 weather

conditions (rainstorm, fog, snow, night, sunny) using flux_edit.py.

Confirmed working: produced 20 real, verified output images.



3DGS camera rendering (scripts folder):



render_cuda_angle.py

Renders the overhead_bridge scene (from yu781986168/3DGS_Mesh_Envs,

MIT license) via the 3DGS_PoseRender CUDA rasterizer, with

command-line camera control (azimuth, elevation, distance, field of

view). Confirmed working, produced angle_filled1.png and other real

renders.



render_cuda_city.py

Same camera logic, pointed at the 2_city scene from the same dataset.

This is the script that produced city_best1.png through

city_best4.png, the actual images used as FLUX input for the full

weather-augmentation demo.



DEPENDENCIES



3DGS_PoseRender (external tool, not included in this repo)

render_cuda_angle.py and render_cuda_city.py require a separate clone

of: https://github.com/guaMass/3DGS_PoseRender



On the DGX, this is cloned at:

home/24jainm/person1_3dgs/renderer/3DGS_PoseRender



Both scripts hardcode this path near the top of the file. Update it

if your clone is somewhere else. Requires a p1_env virtualenv with

torch, plyfile, and the compiled diff_gaussian_rasterization CUDA

extension.



FLUX environment

flux_edit.py requires flux_env, with torch 2.11.0+cu128, diffusers,

transformers, accelerate, and Hugging Face access to

black-forest-labs/FLUX.1-Kontext-dev (gated model, license must be

accepted on huggingface.co).



DATASETS USED



overhead_bridge and 2_city scenes, from yu781986168/3DGS_Mesh_Envs on

Hugging Face (MIT license).



WHAT IS NOT IN THIS REPO (tried, not kept)



A camera script for the Voxel51 truck and train scenes

(render_cuda_generic.py) was written and tested, but those scenes

were dropped as unsuitable for UAV-style imagery (ground-level

capture, not aerial) and are not part of the final demo.



A sky-compositing script for filling black background pixels in low

elevation 3DGS renders was written but never fully tested, and was

not needed for the final city_best1 through city_best4 renders (all

shot from elevations steep enough to avoid the black-sky issue).



CONFIRMED REAL OUTPUT (not synthetic or faked)



All FLUX outputs include a json metadata file recording the exact

model, seed, prompt, and hardware used, per the project's

no-faked-output requirement. All renderer outputs are logged with the

real gaussian count and the renderer's own non-black-fraction

diagnostic to confirm the camera frame was genuinely filled with real

scene data.



