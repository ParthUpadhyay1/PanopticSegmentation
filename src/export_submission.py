from __future__ import annotations
import argparse
import os
from pathlib import Path
import zipfile


def export_zip(pred_dir: str, out_zip: str) -> None:
    pred_dir_p = Path(pred_dir)
    if not pred_dir_p.exists():
        raise FileNotFoundError(f"pred_dir not found: {pred_dir}")

    pngs = sorted([p for p in pred_dir_p.iterdir() if p.suffix.lower() == ".png"])
    if not pngs:
        raise FileNotFoundError(f"No .png predictions found in: {pred_dir}")

    out_zip_p = Path(out_zip)
    out_zip_p.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out_zip_p, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in pngs:
            # IMPORTANT: place files at zip root (no subfolders)
            zf.write(p, arcname=p.name)

    print(f"Created submission zip with {len(pngs)} files -> {out_zip_p}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True, help="Folder containing predicted PNGs")
    ap.add_argument("--out_zip", required=True, help="Output zip path")
    args = ap.parse_args()
    export_zip(args.pred_dir, args.out_zip)


if __name__ == "__main__":
    main()
