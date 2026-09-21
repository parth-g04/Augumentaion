"""Dataset export utilities for Layer 4 augmented observations.
Exports generated frames and annotations into standard COCO JSON and YOLO formats
for downstream model training and validation.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image

from .interfaces import BBox2D


class DatasetExporter:
    """Exports images and bounding box annotations into standard vision datasets."""

    def __init__(self, export_dir: str):
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def export_coco(
        self,
        samples: List[Dict[str, Any]],
        split_name: str = "augmented_train",
        categories: Optional[List[Dict[str, Any]]] = None
    ) -> Path:
        """Exports samples to standard COCO format.
        samples format: list of dicts with keys:
            'id', 'image_path' or 'image_array', 'width', 'height', 'boxes' (list of BBox2D)
        """
        img_dir = self.export_dir / split_name / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        if categories is None:
            categories = [
                {"id": 1, "name": "building", "supercategory": "structure"},
                {"id": 2, "name": "car", "supercategory": "vehicle"},
                {"id": 3, "name": "truck", "supercategory": "vehicle"},
                {"id": 4, "name": "person", "supercategory": "human"}
            ]

        cat_name_to_id = {c["name"]: c["id"] for c in categories}

        images_coco = []
        annotations_coco = []
        ann_id_counter = 1

        for sample in samples:
            img_id = sample["id"]
            img_name = f"{img_id}.png"
            dest_img_path = img_dir / img_name

            # Copy or save image
            if "image_array" in sample:
                Image.fromarray(sample["image_array"]).save(dest_img_path)
            elif "image_path" in sample:
                src_img = Image.open(sample["image_path"])
                src_img.save(dest_img_path)

            W = sample.get("width", 640)
            H = sample.get("height", 480)

            images_coco.append({
                "id": img_id,
                "file_name": img_name,
                "width": W,
                "height": H
            })

            for b in sample.get("boxes", []):
                cat_id = cat_name_to_id.get(b.class_name, 1)
                w = max(0.0, b.xmax - b.xmin)
                h = max(0.0, b.ymax - b.ymin)
                area = w * h

                annotations_coco.append({
                    "id": ann_id_counter,
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": [round(b.xmin, 2), round(b.ymin, 2), round(w, 2), round(h, 2)],
                    "area": round(area, 2),
                    "segmentation": [],
                    "iscrowd": 0
                })
                ann_id_counter += 1

        coco_data = {
            "info": {
                "description": "Layer 4 Generative Augmentation Exported Dataset",
                "version": "1.0",
                "year": 2026
            },
            "licenses": [],
            "images": images_coco,
            "annotations": annotations_coco,
            "categories": categories
        }

        ann_file = self.export_dir / split_name / "annotations.json"
        with open(ann_file, "w", encoding="utf-8") as f:
            json.dump(coco_data, f, indent=2)

        return ann_file

    def export_yolo(
        self,
        samples: List[Dict[str, Any]],
        split_name: str = "augmented_train",
        class_mapping: Optional[Dict[str, int]] = None
    ) -> Path:
        """Exports samples to standard YOLO format: images/ and labels/ with normalized txt files."""
        base_dir = self.export_dir / split_name
        img_dir = base_dir / "images"
        lbl_dir = base_dir / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        mapping = class_mapping or {"building": 0, "car": 1, "truck": 2, "person": 3}

        for sample in samples:
            img_id = sample["id"]
            img_file = img_dir / f"{img_id}.png"
            lbl_file = lbl_dir / f"{img_id}.txt"

            if "image_array" in sample:
                Image.fromarray(sample["image_array"]).save(img_file)
            elif "image_path" in sample:
                Image.open(sample["image_path"]).save(img_file)

            W = float(sample.get("width", 640))
            H = float(sample.get("height", 480))

            lines = []
            for b in sample.get("boxes", []):
                cls_idx = mapping.get(b.class_name, 0)
                # Normalize coordinates for YOLO (center_x, center_y, width, height)
                cx = ((b.xmin + b.xmax) / 2.0) / W
                cy = ((b.ymin + b.ymax) / 2.0) / H
                bw = (b.xmax - b.xmin) / W
                bh = (b.ymax - b.ymin) / H

                lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            with open(lbl_file, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

        return base_dir
