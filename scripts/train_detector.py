#!/usr/bin/env python3
"""Train the single-class speed-limit detector (YOLOv5nu) on TT100K.

Notes:
- imgsz=640 aligns with the deployment input (RDK X5, 640x352); TT100K 2048x2048
  is scaled online by ultralytics — normalized YOLO labels stay consistent.
- Train-time val uses a small fast subset (configs/TT100K_speed_det_trainval.yaml);
  conclusions come from the official test split via scripts/eval_detector_size.py.
- Anchor-free head (yolov5nu): no anchor engineering needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default=str(ROOT / "weights" / "yolov5nu.pt"))
    ap.add_argument("--data", default=str(ROOT / "configs" / "TT100K_speed_det_trainval.yaml"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--name", default="smoke1")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.resume:
        last = ROOT / "runs" / "detector" / args.name / "weights" / "last.pt"
        model = YOLO(str(last))
        model.train(resume=True)
        return 0

    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        seed=0,
        project=str(ROOT / "runs" / "detector"),
        name=args.name,
        exist_ok=True,
        # defaults kept on purpose: mosaic/scale/hsv aug on (helps small objects),
        # fliplr on, flipud off, AMP on
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
