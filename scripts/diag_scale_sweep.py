#!/usr/bin/env python3
"""Multi-scale inference probe: is the _1 confidence notch scale-aliasing?

Diagnosis for reports/frozen_diag: on 20260907_1 (pl60) the frozen detector
LOCALIZES every frame (IoU 0.90-0.97) but its confidence collapses in the
15-20px band (frames 46-50). Run the SAME frames through tt_cropcolor1 at
several input scales; if the notch moves/disappears with input scale, it is a
scale/aliasing artifact, not a scene/appearance effect.

Read-only: no training, no saved weights, no file mutation (predict prints only).

Usage:
  .venv/bin/python scripts/diag_scale_sweep.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SELF_IMG = ROOT / "data" / "self" / "images" / "test"
SELF_DET = ROOT / "data" / "self" / "labels_det" / "test"


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def gt_box(key: str) -> list[float]:
    ln = (SELF_DET / f"{key}.txt").read_text(encoding="utf-8").splitlines()[0].split()
    _, cx, cy, w, h = map(float, ln)
    return [(cx - w / 2) * 640, (cy - h / 2) * 352, (cx + w / 2) * 640, (cy + h / 2) * 352]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="runs/detector/tt_cropcolor1/weights/best.pt")
    ap.add_argument("--trip", default="20260907_1")
    ap.add_argument("--frames", default="44,45,46,47,48,49,50,51,52,53")
    ap.add_argument("--scales", default="0.5,0.75,1.0,1.25,1.5,2.0")
    args = ap.parse_args()

    from ultralytics import YOLO

    frames = [int(x) for x in args.frames.split(",")]
    scales = [float(x) for x in args.scales.split(",")]
    model = YOLO(args.weights)

    imgs = {int(p.stem.rsplit("_", 1)[1]): p
            for p in (SELF_IMG / args.trip).glob("*.png")}
    print(f"trip={args.trip} 权重={args.weights}")
    header = "     帧  尺寸" + "".join(f"{s:>10}" for s in scales)
    print(header)
    for f in frames:
        img = imgs[f]
        key = f"{args.trip}/{img.stem}"
        g = gt_box(key)
        size = min(g[2] - g[0], g[3] - g[1])
        row = f"{f:>7} {size:>5.1f}"
        for s in scales:
            imgsz = [int(round(352 * s)), int(round(640 * s))]
            r = model.predict(source=str(img), imgsz=imgsz, conf=0.0005, iou=0.7,
                              max_det=30, verbose=False, device=0)[0]
            best = 0.0
            if len(r.boxes):
                xy = r.boxes.xyxy.cpu().numpy()
                cf = r.boxes.conf.cpu().numpy()
                for b, c in zip(xy, cf):
                    if iou(g, b.tolist()) >= 0.5:
                        best = max(best, float(c))
            row += f"{best:>10.4f}"
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
