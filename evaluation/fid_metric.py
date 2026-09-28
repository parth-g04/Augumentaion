"""
Dataset-level FID evaluation.

FID must be calculated between two DATASETS/FOLDERS,
not between one original image and one generated image.
"""

import os
from typing import Tuple

import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from PIL import Image
import numpy as np


SUPPORTED_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp"
)


def get_image_files(folder: str):

    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"Image folder not found: {folder}"
        )

    files = []

    for filename in os.listdir(folder):

        if filename.lower().endswith(
            SUPPORTED_EXTENSIONS
        ):

            files.append(
                os.path.join(
                    folder,
                    filename
                )
            )

    files.sort()

    return files


def load_images_as_uint8(
    image_paths
):

    images = []

    for path in image_paths:

        image = Image.open(
            path
        ).convert("RGB")

        image = np.array(
            image
        )

        tensor = torch.from_numpy(
            image
        ).permute(
            2,
            0,
            1
        )

        images.append(
            tensor
        )

    return images


def compute_fid(
    real_dir: str,
    generated_dir: str
) -> Tuple[float, int, int]:

    real_files = get_image_files(
        real_dir
    )

    generated_files = get_image_files(
        generated_dir
    )

    if len(real_files) == 0:

        raise ValueError(
            "No images found in real dataset."
        )

    if len(generated_files) == 0:

        raise ValueError(
            "No images found in generated dataset."
        )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    fid = FrechetInceptionDistance(
        feature=2048,
        normalize=False
    ).to(device)

    # --------------------------------------------------------
    # Real images
    # --------------------------------------------------------

    real_images = load_images_as_uint8(
        real_files
    )

    for image in real_images:

        image = image.unsqueeze(0).to(
            device
        )

        fid.update(
            image,
            real=True
        )

    # --------------------------------------------------------
    # Generated images
    # --------------------------------------------------------

    generated_images = load_images_as_uint8(
        generated_files
    )

    for image in generated_images:

        image = image.unsqueeze(0).to(
            device
        )

        fid.update(
            image,
            real=False
        )

    score = fid.compute()

    return (
        float(score.item()),
        len(real_files),
        len(generated_files)
    )