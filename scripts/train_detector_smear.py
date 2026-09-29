#!/usr/bin/env python3
"""One approved TT100K-only detector ablation: post-resize local resampling smear.

All other smoke1 training arguments stay the same (weights, split, optimizer defaults,
epochs, image size, batch, seed). The train-only transform is inserted after the
Ultralytics spatial pipeline at 640x640 and before Format. A local anisotropic
downsample/upscale loses fine detail while returning every pixel to its ORIGINAL
coordinate; detection boxes remain unchanged. Val/test are not transformed.

    .venv/bin/python scripts/train_detector_smear.py
"""

from __future__ import annotations

import argparse
import random
from copy import copy
from pathlib import Path

import cv2
import numpy as np
import ultralytics
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import get_split_fraction
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import colorstr
from ultralytics.utils.torch_utils import unwrap_model

ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_VERSION = "8.4.157"  # custom dataset matches build_yolo_dataset in this release


class LocalResampleSmear:
    """Train-only patch compression/reconstruction with smoothly blended boundaries.

    No persistent warp or new scene geometry: a frame/object stays in place, so
    Instances and YOLO labels MUST stay bit-identical. Half of frames remain clean.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    @staticmethod
    def centers(labels: dict, w: int, h: int) -> np.ndarray:
        ins = labels.get("instances")
        if ins is None or len(ins) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        boxes = ins.bboxes  # read-only; never mutate Ultralytics Instances
        fmt = ins._bboxes.format
        if fmt == "xyxy":
            xy = (boxes[:, :2] + boxes[:, 2:]) / 2
        elif fmt == "xywh":
            xy = boxes[:, :2].copy()
        elif fmt == "ltwh":
            xy = boxes[:, :2] + boxes[:, 2:] / 2
        else:
            raise ValueError(f"unexpected bbox format {fmt!r}")
        if ins.normalized:
            xy *= np.array([w, h])
        return xy

    def __call__(self, labels: dict) -> dict:
        if random.random() >= self.p:
            return labels
        img = labels["img"]  # BGR uint8, spatial augmentation already done
        h, w = img.shape[:2]
        candidates = self.centers(labels, w, h)
        if len(candidates) and random.random() < 0.8:
            cx, cy = candidates[random.randrange(len(candidates))]
        else:
            cx, cy = random.uniform(0, w), random.uniform(0, h)
        cx, cy = float(np.clip(cx, 0, w - 1)), float(np.clip(cy, 0, h - 1))
        rx, ry = random.uniform(0.16, 0.34) * w, random.uniform(0.16, 0.34) * h
        x1, x2 = max(0, int(cx - rx)), min(w, int(cx + rx))
        y1, y2 = max(0, int(cy - ry)), min(h, int(cy + ry))
        pw, ph = x2 - x1, y2 - y1
        if pw < 8 or ph < 8:
            return labels

        strong = random.uniform(0.55, 0.85)
        weak = random.uniform(0.82, 1.0)
        sx, sy = (strong, weak) if random.random() < 0.5 else (weak, strong)
        region = img[y1:y2, x1:x2]
        smaller = cv2.resize(region, (max(2, round(pw * sx)), max(2, round(ph * sy))),
                             interpolation=cv2.INTER_AREA)
        restored = cv2.resize(smaller, (pw, ph), interpolation=cv2.INTER_LINEAR)

        # Smooth blend around the *interior* of the ROI, but keep strength near
        # the frame edge where GS130WI signs often appear. No paste seams.
        xx = np.arange(x1, x2, dtype=np.float32)[None, :]
        yy = np.arange(y1, y2, dtype=np.float32)[:, None]
        gaussian = np.exp(-0.5 * (((xx - cx) / max(1, rx * 0.65)) ** 2 +
                                  ((yy - cy) / max(1, ry * 0.65)) ** 2))
        left = 1.0 if x1 == 0 else np.clip((xx - x1) / 24, 0, 1)
        right = 1.0 if x2 == w else np.clip((x2 - 1 - xx) / 24, 0, 1)
        top = 1.0 if y1 == 0 else np.clip((yy - y1) / 24, 0, 1)
        bottom = 1.0 if y2 == h else np.clip((y2 - 1 - yy) / 24, 0, 1)
        alpha = (0.85 * gaussian * left * right * top * bottom)[..., None]
        img[y1:y2, x1:x2] = np.clip(region * (1 - alpha) + restored * alpha, 0, 255).astype(np.uint8)
        return labels


class SmearYOLODataset(YOLODataset):
    """Rebuildable train-only transform (including after close_mosaic at epoch 90)."""

    def build_transforms(self, hyp=None):
        transforms = super().build_transforms(hyp)
        if self.augment:
            transforms.insert(-1, LocalResampleSmear())
        return transforms


class SmearDetectionTrainer(DetectionTrainer):
    """Keep upstream data/validation behavior; only replace train dataset type."""

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        if mode != "train":
            return super().build_dataset(img_path, mode, batch)
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        return SmearYOLODataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=True,
            hyp=copy(self.args),
            rect=self.args.rect,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=gs,
            pad=0.0,
            prefix=colorstr("train: "),
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=get_split_fraction(self.args.fraction, mode),
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default=str(ROOT / "weights" / "yolov5nu.pt"))
    ap.add_argument("--data", default=str(ROOT / "configs" / "TT100K_speed_det_trainval.yaml"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--fraction", type=float, default=1.0, help="smoke-only; full run must use 1.0")
    ap.add_argument("--name", default="tt_smear1")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if ultralytics.__version__ != ULTRALYTICS_VERSION:
        raise SystemExit(f"review build_yolo_dataset compatibility: expected {ULTRALYTICS_VERSION}, "
                         f"got {ultralytics.__version__}")
    run = ROOT / "runs" / "detector" / args.name
    if run.exists() and not args.resume:
        raise SystemExit(f"{run} already exists; choose another --name or --resume")

    if args.resume:
        model = YOLO(str(run / "weights" / "last.pt"))
        model.train(trainer=SmearDetectionTrainer, resume=True)
    else:
        model = YOLO(args.weights)
        model.train(
            trainer=SmearDetectionTrainer,
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            patience=args.patience,
            fraction=args.fraction,
            seed=0,
            project=str(ROOT / "runs" / "detector"),
            name=args.name,
            exist_ok=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
