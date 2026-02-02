from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm
import torch

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog

from src.utils.config import load_yaml_with_includes
from src.data.register_lars import register_from_cfg, THING_IDS, STUFF_IDS
from src.utils.panoptic_encoding import encode_panoptic_rgb


def _list_images(folder: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = []
    for p in Path(folder).iterdir():
        if p.suffix.lower() in exts:
            paths.append(str(p))
    return sorted(paths)


def _invert_map(d: Dict[int, int]) -> Dict[int, int]:
    return {v: k for k, v in d.items()}


def run_inference(cfg_dict: Dict[str, Any]) -> None:
    # paths = cfg_dict.get("paths", {})
    paths = {**cfg_dict, **cfg_dict.get("paths", {})}
    infer = cfg_dict["infer"]

    # Register train/val for metadata; test has no GT but uses same class definitions
    register_from_cfg(paths, cache_dir="artifacts/d2_cache")
    meta = MetadataCatalog.get("lars_val_panoptic_separated")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(infer["d2_base_config"]))
    cfg.MODEL.WEIGHTS = infer["weights"]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(THING_IDS)
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(STUFF_IDS)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(infer.get("score_thresh", 0.5))
    cfg.MODEL.DEVICE = infer.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    predictor = DefaultPredictor(cfg)

    out_dir = Path(infer["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    test_images = _list_images(paths["images_test"])
    if not test_images:
        raise FileNotFoundError(f"No images found in images_test: {paths['images_test']}")
    print(f"Found {len(test_images)} test images")

    thing_contig_to_dataset = _invert_map(getattr(meta, "thing_dataset_id_to_contiguous_id"))
    stuff_contig_to_dataset = _invert_map(getattr(meta, "stuff_dataset_id_to_contiguous_id"))

    for img_path in tqdm(test_images, desc="Infer"):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        outputs = predictor(img)

        # Detectron2 panoptic output
        panoptic_seg, segments_info = outputs["panoptic_seg"]
        pan_id = panoptic_seg.cpu().numpy().astype(np.int32)

        class_map = np.zeros_like(pan_id, dtype=np.uint8)
        instance_map = np.zeros_like(pan_id, dtype=np.int32)

        next_inst_id = 1
        for seg in segments_info:
            seg_mask = (pan_id == int(seg["id"]))
            cat_contig = int(seg["category_id"])
            isthing = bool(seg.get("isthing", False))
            if isthing:
                class_id = int(thing_contig_to_dataset[cat_contig])
                inst_id = next_inst_id
                next_inst_id += 1
            else:
                class_id = int(stuff_contig_to_dataset[cat_contig])
                inst_id = 0
            class_map[seg_mask] = np.uint8(class_id)
            instance_map[seg_mask] = np.int32(inst_id)

        rgb = encode_panoptic_rgb(class_map, instance_map)

        out_path = out_dir / (Path(img_path).stem + ".png")
        cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    print(f"Wrote predictions to: {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg_dict = load_yaml_with_includes(args.config)
    run_inference(cfg_dict)


if __name__ == "__main__":
    main()
