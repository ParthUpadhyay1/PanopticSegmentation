from __future__ import annotations
import argparse
import os
from typing import Any, Dict

import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_setup, launch
from detectron2 import model_zoo

from detectron2.data import build_detection_train_loader
from detectron2.data.dataset_mapper import DatasetMapper
# from detectron2.data.dataset_mapper import PanopticDatasetMapper
from detectron2.data import build_detection_test_loader


from src.utils.config import load_yaml_with_includes
from src.data.register_lars import register_from_cfg, THING_IDS, STUFF_IDS


class Trainer(DefaultTrainer):
    """DefaultTrainer works well for baseline. You can customize hooks/mappers later."""
    @classmethod
    def build_train_loader(cls, cfg):
        # Use panoptic mapper (reads panoptic PNG + segments_info)
        # mapper = PanopticDatasetMapper(cfg, is_train=True)
        # return build_detection_train_loader(cfg, mapper=mapper)
        return build_detection_train_loader(cfg)
    
    # @classmethod
    # def build_test_loader(cls, cfg, dataset_name):
    #     # mapper = PanopticDatasetMapper(cfg, is_train=False)
    #     # return build_detection_test_loader(cfg, dataset_name, mapper=mapper)
    #     pass


def setup(cfg_dict: Dict[str, Any]):
    # paths = cfg_dict.get("paths", {})
    paths = {**cfg_dict, **cfg_dict.get("paths", {})}
    train = cfg_dict["train"]

    register_from_cfg(paths, cache_dir=os.path.join(train["output_dir"], "d2_cache"))

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(train["d2_base_config"]))
    cfg.INPUT.MASK_FORMAT = "bitmask"
    if hasattr(cfg.INPUT, "INSTANCE_MASK_FORMAT"):
        cfg.INPUT.INSTANCE_MASK_FORMAT = "bitmask"

    cfg.DATASETS.TRAIN = ("lars_train_panoptic_separated",)
    cfg.DATASETS.TEST = ("lars_val_panoptic_separated",)

    cfg.DATALOADER.NUM_WORKERS = int(train["num_workers"])
    cfg.SOLVER.IMS_PER_BATCH = int(train["ims_per_batch"])
    cfg.SOLVER.BASE_LR = float(train["base_lr"])
    cfg.SOLVER.MAX_ITER = int(train["max_iter"])
    cfg.SOLVER.WARMUP_ITERS = int(train["warmup_iters"])
    cfg.TEST.EVAL_PERIOD = int(train["eval_period"])
    cfg.SOLVER.CHECKPOINT_PERIOD = int(train["checkpoint_period"])
    # cfg.SOLVER.AMP.ENABLED = True
    # if hasattr(cfg.SOLVER, "AMP"):
    #     cfg.SOLVER.AMP.ENABLED = True



    # Resize
    cfg.INPUT.MIN_SIZE_TRAIN = train["min_size_train"]
    cfg.INPUT.MAX_SIZE_TRAIN = int(train["max_size_train"])
    cfg.INPUT.MIN_SIZE_TEST = int(train["min_size_test"])
    cfg.INPUT.MAX_SIZE_TEST = int(train["max_size_test"])

    # Num classes
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(THING_IDS)   # 8 thing classes
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(STUFF_IDS)  # 3 stuff classes

    # Init from COCO pretrained weights
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(train["d2_base_config"])

    cfg.OUTPUT_DIR = train["output_dir"]
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Device
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # cfg.MODEL.DEVICE = "cpu"

    default_setup(cfg, args=None)
    cfg.freeze()
    return cfg, bool(train.get("resume", True))


def main_worker(args):
    cfg_dict = load_yaml_with_includes(args.config)
    cfg, resume = setup(cfg_dict)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=resume)
    return trainer.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config in ./configs")
    ap.add_argument("opts", nargs=argparse.REMAINDER, help="Optional detectron2 cfg overrides (advanced)")
    args = ap.parse_args()

    # If user passed detectron2-style overrides, apply them after loading YAML.
    # Example: python -m src.train --config ... SOLVER.BASE_LR 0.0001
    # For simplicity, we recommend editing YAML; but this is still supported.
    if args.opts:
        # We'll apply overrides by patching after setup.
        pass

    # Single machine launch
    launch(
        main_worker,
        num_gpus_per_machine=1,
        num_machines=1,
        machine_rank=0,
        dist_url="auto",
        args=(args,),
    )


if __name__ == "__main__":
    main()
