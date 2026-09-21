# Layer 4: Generative Augmentation Engine

This repository contains the implementation of **Layer 4: Generative Augmentation & Sensor Domain Adaptation Engine** for the Drone Data Augmentation Pipeline (IIT Mandi / IIT Bombay collaboration).

## Core Principle: "One Flight → Infinite Data"
```
[ 3D Scene (.ply / surrogate) ]
               │
     ┌─────────┴─────────┐
Camera Pose 1       Camera Pose 2
     │                   │
RGB-D View 1        RGB-D View 2
     │                   │
  ┌──┴──┐             ┌──┴──┐
Rain   Fog          Night  Sunset
```
- **3DGS / Scene Reconstruction** (Layers 1–3) handles *geometric & viewpoint diversity*.
- **Generative Augmentation Engine** (Layer 4) handles *appearance, illumination, adverse weather, and sensor domain shifts*.

## Hardware & Execution Split
- **Local Machine**: Verified NVIDIA GeForce RTX 4050 Laptop GPU (6141 MiB VRAM), Driver 592.00.
  - Used for 6-DOF camera trajectory sweeps, structural conditioning (depth, semantic masks, edges), fast baseline generative adapters, and the Quality Gate evaluation suite.
- **DGX B200 Cluster**: 8x B200 GPUs (1,440 GB VRAM).
  - Used for large-scale flow matching (`FLUX.1 Kontext` / `SD3.5`), high-resolution domain transfer sweeps, and dataset-scale FID benchmarking.

## Directory Structure
- `scene_3dgs/`: 6-DOF camera trajectory generator and decoupled `SceneRenderer` interface.
- `conditioning/`: Structural conditioning engine (depth provenance, semantic segmentation, Canny/Sobel edges).
- `generation/`: Flow-based and latent generative augmentation models.
- `evaluation/`: Multi-metric Quality Gate (bounding box consistency, count drift, edge similarity, PASS/REJECT).
- `experiments/`: Experiment configurations (`configs/EXP*.yaml`) and execution runner.
- `tests/`: Automated verification test suite.
