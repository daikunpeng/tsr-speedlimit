#!/usr/bin/env python3
"""Create reversible GS130WI inference-only input A/B images under runs/.

Original data/self images/labels stay untouched. This is a diagnostic for frozen
detector/classifier weights, not a training-data generator or a restoration claim.

    .venv/bin/python scripts/prepare_self_preproc.py --variant sharp
    .venv/bin/python scripts/prepare_self_preproc.py --variant clahe

Pass --image-root runs/preproc/<variant>/images/test to eval_e2e.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUFFIXES = {".png", ".jpg", ".jpeg"}


def preprocess(img, variant: str):
    import cv2

    if variant == "sharp":
        blur = cv2.GaussianBlur(img, (0, 0), sigmaX=0.7)
        return cv2.addWeighted(img, 1.35, blur, -0.35, 0)
    if variant == "clahe":
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    raise ValueError(variant)


def main() -> int:
    import cv2

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", required=True, choices=("sharp", "clahe"))
    ap.add_argument("--src", default="data/self/images/test")
    ap.add_argument("--name", default=None, help="output name under runs/preproc (default: variant)")
    args = ap.parse_args()

    src = ROOT / args.src
    images = sorted(p for p in src.rglob("*") if p.suffix.lower() in SUFFIXES)
    if not images:
        raise SystemExit(f"no images under {src}")
    name = args.name or args.variant
    out = ROOT / "runs" / "preproc" / name
    if out.exists():
        raise SystemExit(f"{out} exists; choose a new --name to preserve prior A/B results")

    for i, ip in enumerate(images, 1):
        img = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        if img is None or img.shape[:2] != (352, 640):
            raise SystemExit(f"cannot read native 640x352 image: {ip}")
        dst = out / "images" / "test" / ip.relative_to(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        opts = [cv2.IMWRITE_PNG_COMPRESSION, 1] if dst.suffix.lower() == ".png" else [cv2.IMWRITE_JPEG_QUALITY, 95]
        if not cv2.imwrite(str(dst), preprocess(img, args.variant), opts):
            raise SystemExit(f"failed to write {dst}")
        if i % 200 == 0:
            print(f"processed {i}/{len(images)}", flush=True)
    (out / "recipe.json").write_text(json.dumps({
        "variant": args.variant, "source": args.src, "n_images": len(images),
        "input_size": [640, 352],
        "params": ({"unsharp_sigma": 0.7, "weight": 0.35} if args.variant == "sharp"
                   else {"lab_clip_limit": 1.5, "tile_grid": [8, 8]}),
    }, indent=2), encoding="utf-8")
    print(f"wrote {len(images)} processed frames to {out / 'images/test'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
