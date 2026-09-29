#!/usr/bin/env python3
"""One approved TT100K-only detector ablation: plate-local color/exposure aug on
top of the tt_crop1 640x352 crop recipe (A->B plan step B, user 2026-09-24).

Reuses CropWide352 from scripts/train_detector_crop352.py and adds a train-only
PlateColorAug transform before Format: with per-box probability each GT sign
region is perturbed (dark / washed-out / brightness-saturation-gamma jitter),
targeting the self-domain misses quantified in reports/diag_missed_frames.md
(dark desaturated red rings, overexposed centers). HUE is never shifted, so
pl(white/red) vs il(blue) and pm(weight) negatives stay separable. Box geometry
is untouched -> no label re-sync. Val/test stay unchanged; no self images used.

    .venv/bin/python scripts/train_detector_crop_color.py
"""

from __future__ import annotations

import argparse
import random
from copy import copy
from importlib.util import module_from_spec, spec_from_file_location
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

spec = spec_from_file_location("crop352", ROOT / "scripts" / "train_detector_crop352.py")
crop_mod = module_from_spec(spec)
spec.loader.exec_module(crop_mod)
CropWide352 = crop_mod.CropWide352


_worker_rng = None


def _aug_rng():
    global _worker_rng
    if _worker_rng is None:
        import torch
        _worker_rng = random.Random(torch.initial_seed() % (2 ** 32))
    return _worker_rng


def plate_color_augment(img: np.ndarray, rng) -> np.ndarray:
    """Plate-local color/exposure jitter on a BGR uint8 region (train only).

    Same modes as the classifier v3 (scripts/train_classifier.py::color_augment)
    but on BGR detector image regions: dark red / washed-out / generic jitter.
    Hue untouched.
    """
    import cv2
    r = rng.random()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    if r < 0.4:  # dark red / low light
        hsv[..., 1] *= rng.uniform(0.35, 0.80)
        hsv[..., 2] *= rng.uniform(0.40, 0.75)
        x = np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32), 0, 255)
        x = 255.0 * (x / 255.0) ** rng.uniform(1.0, 1.6)
    elif r < 0.7:  # overexposed / washed-out
        x = img.astype(np.float32) * rng.uniform(0.50, 0.75) + rng.uniform(70.0, 115.0)
        hsv2 = cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv2[..., 1] *= rng.uniform(0.45, 0.85)
        x = np.clip(cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR).astype(np.float32), 0, 255)
    else:  # generic brightness / saturation / gamma jitter
        hsv[..., 1] *= rng.uniform(0.70, 1.30)
        hsv[..., 2] *= rng.uniform(0.70, 1.30)
        x = np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32), 0, 255)
        x = 255.0 * (x / 255.0) ** rng.uniform(0.80, 1.25)
    return np.clip(x, 0, 255).astype(np.uint8)


def _feather(bw: int, bh: int) -> np.ndarray:
    """Radial falloff mask (1 at center, ~0 at edge) to avoid hard seams."""
    yy, xx = np.mgrid[0:bh, 0:bw].astype(np.float32)
    cx, cy = (bw - 1) / 2, (bh - 1) / 2
    d = np.sqrt(((xx - cx) / max(1, bw / 2)) ** 2 + ((yy - cy) / max(1, bh / 2)) ** 2)
    return np.clip(1.4 - 1.6 * d, 0.0, 1.0).astype(np.float32)


class PlateColorAug:
    """Per-GT-box plate color/exposure jitter; boxes unchanged, no cls re-sync."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, labels: dict) -> dict:
        if self.p <= 0:
            return labels
        ins = labels.get("instances")
        if ins is None or not len(ins):
            return labels
        img = labels["img"]
        H, W = img.shape[:2]
        fmt = ins._bboxes.format
        bb = ins.bboxes
        if fmt == "xyxy":
            boxes = bb
        elif fmt == "xywh":
            boxes = np.stack([bb[:, 0] - bb[:, 2] / 2, bb[:, 1] - bb[:, 3] / 2,
                              bb[:, 0] + bb[:, 2] / 2, bb[:, 1] + bb[:, 3] / 2], 1)
        elif fmt == "ltwh":
            boxes = np.stack([bb[:, 0], bb[:, 1], bb[:, 0] + bb[:, 2], bb[:, 1] + bb[:, 3]], 1)
        else:
            raise ValueError(f"unexpected bbox format {fmt!r}")
        rng = _aug_rng()
        for box in boxes:
            if rng.random() >= self.p:
                continue
            x1, y1 = max(0, int(round(box[0]))), max(0, int(round(box[1])))
            x2, y2 = min(W, int(round(box[2]))), min(H, int(round(box[3])))
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            region = img[y1:y2, x1:x2]
            aug = plate_color_augment(region, rng)
            m = _feather(x2 - x1, y2 - y1)[..., None]
            img[y1:y2, x1:x2] = np.clip(region * (1 - m) + aug * m, 0, 255).astype(np.uint8)
        return labels


class CropColorYOLODataset(YOLODataset):
    def build_transforms(self, hyp=None):
        transforms = super().build_transforms(hyp)
        if self.augment:
            transforms.insert(-1, CropWide352())
            transforms.insert(-1, PlateColorAug())
        return transforms


class CropColorTrainer(DetectionTrainer):
    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        if mode != "train":
            return super().build_dataset(img_path, mode, batch)
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        return CropColorYOLODataset(
            img_path=img_path, imgsz=self.args.imgsz, batch_size=batch, augment=True,
            hyp=copy(self.args), rect=self.args.rect, cache=self.args.cache or None,
            single_cls=self.args.single_cls or False, stride=gs, pad=0.0,
            prefix=colorstr("train: "), task=self.args.task, classes=self.args.classes,
            data=self.data, fraction=get_split_fraction(self.args.fraction, mode),
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
    ap.add_argument("--multi-scale", type=float, default=0.33,
                    help="multiscale training range as fraction of imgsz (0 disables)")
    ap.add_argument("--name", default="tt_cropcolor1")
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
        model.train(trainer=CropColorTrainer, resume=True)
    else:
        model = YOLO(args.weights)
        model.train(
            trainer=CropColorTrainer, data=args.data, epochs=args.epochs, imgsz=args.imgsz,
            batch=args.batch, workers=args.workers, patience=args.patience,
            fraction=args.fraction, seed=0, project=str(ROOT / "runs" / "detector"),
            name=args.name, exist_ok=True, multi_scale=args.multi_scale,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
