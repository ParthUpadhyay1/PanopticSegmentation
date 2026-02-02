# USV Panoptic Segmentation (LaRS) — Detectron2 baseline

This is a **clean, runnable starter project** for the MaCVi USV-based Panoptic Segmentation track (LaRS dataset).

It trains a Detectron2 panoptic model (PanopticFPN) on LaRS and exports predictions in the **challenge PNG encoding**:

- **R** = class id  
- **G,B** = instance id (`instance_id = G*256 + B`)  
- Stuff instances are commonly set to **0**.

## 0) What you need to download

From the LaRS dataset page, download:

- **Images** (train/val/test)
- **Annotations** (train/val COCO-panoptic JSON + panoptic masks)

You should end up with a folder layout like:

```
/path/to/LaRS/
  images/
    train/   (*.jpg or *.png)
    val/
    test/
  annotations/
    panoptic_annotations_train.json
    panoptic_annotations_val.json
  panoptic_masks/
    train/   (*.png)
    val/
```

> The exact filenames may differ depending on the dataset version you downloaded.  
> This project is flexible: you set paths in `configs/local_paths.yaml`.

## 1) Install

Recommended: create a fresh env (Python 3.10/3.11 recommended).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip

# Install torch first (pick the CUDA build that matches your machine)
# Example (CUDA 12.1):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

Detectron2 installation varies by OS/CUDA. If `pip install detectron2` fails, follow Detectron2's official install instructions.

## 2) Set dataset paths

Copy and edit:

```bash
cp configs/local_paths.example.yaml configs/local_paths.yaml
```

Then update the paths inside `configs/local_paths.yaml`.

## 3) Train

```bash
python -m src.train --config configs/train_panoptic_fpn.yaml
```

Outputs (checkpoints, logs) go to `artifacts/`.

## 4) Run inference on test split & create submission zip

```bash
python -m src.infer --config configs/infer.yaml
python -m src.export_submission --pred_dir artifacts/preds_test --out_zip artifacts/submission.zip
```

The submission zip will contain **PNG files in the root of the zip** (no subfolders), matching the test image filenames.

## 5) Notes / Tips

- Start with short training (few thousand iters) to validate the pipeline.
- You can later swap the architecture (Mask2Former, MaskDINO, etc.) while keeping
  the dataset + export-format code.

## References

- MaCVi USV Panoptic Segmentation rules (PNG encoding, class ids)
- LaRS dataset (ICCV 2023)
