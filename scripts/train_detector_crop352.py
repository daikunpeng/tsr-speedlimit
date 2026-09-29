#!/usr/bin/env python3
"""One approved TT100K-only detector ablation: online 640x352 position crop.

User-approved follow-up to reports/detector_smear_ablation.md (plan confirmed
2026-09-24). All other smoke1 training arguments stay the same (weights, split,
epochs, imgsz=640 square pipeline, batch, seed). After the standard Ultralytics
spatial pipeline (mosaic/affine/flips at 640x640) and before Format, every
training image is cropped to a 640x352 window — matching the deployment frame
(RDK X5 input). Bounding boxes are shifted and clipped with the crop; boxes
outside the window are dropped together with their class entries (same
cls/instance synchronization as RandomPerspective.apply_instances).

Window placement (y0 in [0, 288]):
  - p_top: window chosen so a random GT sign center lands in the top
    5-30% of the 352px window (self 24-48px signs: 79% at top vs TT100K 8%).
  - otherwise uniform y0.

Val/test stay untouched (square letterbox). No self images are used for training.

    .venv/bin/python scripts/train_detector_crop352.py
"""

from __future__ import annotations

import argparse
import random
from copy import copy
from pathlib import Path

import numpy as np
import ultralytics
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import get_split_fraction
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import colorstr
from ultralytics.utils.torch_utils import unwrap_model

ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_VERSION = "8.4.157"


class CropWide352:
    """Train-only 640x352 window crop with synchronized label updates."""

    def __init__(self, height: int = 352, p_top: float = 0.5, top_band=(0.05, 0.30)):
        self.height = height
        self.p_top = p_top
        self.top_band = top_band

    def _choose_y0(self, img_h: int, boxes_xyxy: np.ndarray | None) -> int:
        max_y0 = img_h - self.height
        if boxes_xyxy is not None and len(boxes_xyxy) and random.random() < self.p_top:
            i = random.randrange(len(boxes_xyxy))
            cy = (boxes_xyxy[i][1] + boxes_xyxy[i][3]) / 2
            target = random.uniform(*self.top_band) * self.height
            return int(np.clip(round(cy - target), 0, max_y0))
        return random.randint(0, max_y0)

    def __call__(self, labels: dict) -> dict:
        img = labels["img"]
        img_h, img_w = img.shape[:2]
        if img_h <= self.height:
            return labels
        ins = labels.get("instances")
        cls = labels.get("cls")

        boxes_xyxy = None
        if ins is not None and len(ins):
            fmt = ins._bboxes.format
            bb = ins.bboxes
            if fmt == "xyxy":
                boxes_xyxy = bb
            elif fmt == "xywh":
                boxes_xyxy = np.stack([bb[:, 0] - bb[:, 2] / 2, bb[:, 1] - bb[:, 3] / 2,
                                       bb[:, 0] + bb[:, 2] / 2, bb[:, 1] + bb[:, 3] / 2], 1)
            elif fmt == "ltwh":
                boxes_xyxy = np.stack([bb[:, 0], bb[:, 1], bb[:, 0] + bb[:, 2],
                                       bb[:, 1] + bb[:, 3]], 1)
            else:
                raise ValueError(f"unexpected bbox format {fmt!r}")

        y0 = self._choose_y0(img_h, boxes_xyxy)
        labels["img"] = img[y0:y0 + self.height]

        if ins is None or not len(ins):
            return labels
        ori_fmt = ins._bboxes.format
        ins.convert_bbox("xyxy")
        bb = ins.bboxes  # stored array; in-place shift below
        bb[:, [1, 3]] -= y0
        segs = ins.segments
        if segs is not None and len(segs):
            segs[..., 1] -= y0
        ins.clip(img_w, self.height)
        good = ins.remove_zero_area_boxes()
        if not all(good) and cls is not None and len(cls) == len(good):
            labels["cls"] = cls[good]
        ins.convert_bbox(ori_fmt)
        return labels


class CropYOLODataset(YOLODataset):
    """Rebuildable train-only transform (including after close_mosaic)."""

    def build_transforms(self, hyp=None):
        transforms = super().build_transforms(hyp)
        if self.augment:
            transforms.insert(-1, CropWide352())
        return transforms


class CropDetectionTrainer(DetectionTrainer):
    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        if mode != "train":
            return super().build_dataset(img_path, mode, batch)
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        return CropYOLODataset(
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
    ap.add_argument("--name", default="tt_crop1")
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
        model.train(trainer=CropDetectionTrainer, resume=True)
    else:
        model = YOLO(args.weights)
        model.train(
            trainer=CropDetectionTrainer,
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
