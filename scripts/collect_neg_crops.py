#!/usr/bin/env python3
"""Collect negative crops (`other` class material) for classifier v2 (C plan).

Sources:
  1. label  : pm*/pr*/ph* boxes from ORIGINAL TT100K labels, cropped at
               scales 1.0/1.3/1.5/2.0 (same generator as speed crops).
  2. detfp  : detector FPs (conf>=0.5, unmatched vs speed GT) on train/val,
               cropped with +20% pad — exactly the e2e deployment crop path.
               test FPs are re-cropped from runs/eval/e2e_test/e2e_raw.json
               (eval-only negatives; never trained on).

Output:
  data/cls_neg/manifest.csv   columns: crop_relpath, old_name('other'), split,
                             scale, sign_w, sign_h, source_image, origin, src_class, det_conf
  data/cls_neg/images/{detfp,label}/<split>/...

Splits: train/val rows are TRAINING negatives; test rows are EVAL-only
(classifier v2 must reject them as `other`).
"""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "TT100K"
DET = ROOT / "data" / "TT100K_speed_det"
OUT = ROOT / "data" / "cls_neg"
E2E_RAW = ROOT / "runs" / "eval" / "e2e_test" / "e2e_raw.json"
SPLITS = ("train", "val", "test")
SCALES = (1.0, 1.3, 1.5, 2.0)
PREFIXES = ("pm", "pr", "ph")
PAD = 0.20
DET_CONF = 0.50
IOU = 0.5
IMG_EXT = (".jpg", ".jpeg", ".png")

MANIFEST_FIELDS = ["crop_relpath", "old_name", "split", "scale", "sign_w", "sign_h",
                   "source_image", "origin", "src_class", "det_conf"]


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    s = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / s if s > 0 else 0.0


def find_image(split: str, stem: str) -> Path | None:
    for ext in IMG_EXT:
        p = SRC / "images" / split / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def crop_pad_save(img: np.ndarray, box, path: Path) -> bool:
    """+PAD crop like eval_e2e deployment path; save original pixels."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * PAD, (y2 - y1) * PAD
    cx1, cy1 = max(0, int(x1 - px)), max(0, int(y1 - py))
    cx2, cy2 = min(w, int(x2 + px)), min(h, int(y2 + py))
    if cx2 - cx1 < 4 or cy2 - cy1 < 4:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img[cy1:cy2, cx1:cx2], [cv2.IMWRITE_JPEG_QUALITY, 95])
    return True


def collect_label_negs(names: dict[int, str], rows: list[dict]) -> dict[str, int]:
    spec = spec_from_file_location("cs", ROOT / "scripts" / "crop_speed_signs.py")
    cs = module_from_spec(spec)
    spec.loader.exec_module(cs)
    counts = {"pm": 0, "pr": 0, "ph": 0}
    for split in SPLITS:
        lbl_dir = SRC / "labels" / split
        for lf in sorted(lbl_dir.glob("*.txt")):
            img_path = find_image(split, lf.stem)
            if img_path is None:
                continue
            boxes = []
            for line in lf.read_text(encoding="utf-8").splitlines():
                p = line.split()
                if len(p) != 5:
                    continue
                cid = int(float(p[0]))
                name = names.get(cid, "")
                if not name.startswith(PREFIXES):
                    continue
                cx, cy, bw, bh = map(float, p[1:5])
                boxes.append((name, cx * 2048, cy * 2048, bw * 2048, bh * 2048))
            if not boxes:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            H, W = img.shape[:2]
            for i, (name, x1, y1, bw, bh) in enumerate(boxes):
                counts[name[:2]] += 1
                for scale in SCALES:
                    bx, by, cw, ch = cs.expand_box(x1, y1, bw, bh, scale, W, H)
                    if cw < 4 or ch < 4:
                        continue
                    tag = cs.scale_tag(scale)
                    rel = f"images/label/{split}/{lf.stem}_i{i:02d}_{name}_{tag}.jpg"
                    (OUT / rel).parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(OUT / rel), img[by:by + ch, bx:bx + cw],
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                    rows.append({
                        "crop_relpath": rel, "old_name": "other", "split": split,
                        "scale": scale, "sign_w": round(bw, 1), "sign_h": round(bh, 1),
                        "source_image": f"{lf.stem}.jpg", "origin": "label",
                        "src_class": name, "det_conf": "",
                    })
    return counts


def load_gt_boxes(split: str) -> dict[str, list[list[float]]]:
    gts: dict[str, list[list[float]]] = {}
    for lf in (DET / "labels" / split).glob("*.txt"):
        boxes = []
        for line in lf.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) != 5:
                continue
            _, cx, cy, w, h = map(float, p)
            boxes.append([(cx - w / 2) * 2048, (cy - h / 2) * 2048,
                          (cx + w / 2) * 2048, (cy + h / 2) * 2048])
        gts[lf.stem] = boxes
    return gts


def collect_detfp_train_val(rows: list[dict], det_weights: Path) -> tuple[int, int]:
    """Run detector on train+val, keep unmatched preds as FP crops."""
    from ultralytics import YOLO
    det = YOLO(str(det_weights))
    n_fp = n_img = 0
    for split in ("train", "val"):
        gts = load_gt_boxes(split)
        img_dir = DET / "images" / split
        results = det.predict(source=str(img_dir), imgsz=640, conf=DET_CONF,
                              iou=0.7, max_det=100, stream=True, verbose=False, device=0)
        t0 = time.time()
        for r in results:
            stem = Path(r.path).stem
            gt_boxes = gts.get(stem, [])
            if not len(r.boxes):
                continue
            n_img += 1
            xyxy = r.boxes.xyxy.cpu().numpy().tolist()
            conf = r.boxes.conf.cpu().numpy().tolist()
            img = None
            used: set[int] = set()
            for pi in sorted(range(len(xyxy)), key=lambda i: -conf[i]):
                best_g, best_v = -1, IOU
                for gi, g in enumerate(gt_boxes):
                    if gi in used:
                        continue
                    v = iou(xyxy[pi], g)
                    if v > best_v:
                        best_g, best_v = gi, v
                if best_g >= 0:
                    used.add(best_g)
                    continue
                # FP -> crop
                if img is None:
                    img = cv2.imread(r.path)
                    if img is None:
                        break
                rel = f"images/detfp/{split}/{stem}_p{pi:02d}.jpg"
                if crop_pad_save(img, xyxy[pi], OUT / rel):
                    rows.append({
                        "crop_relpath": rel, "old_name": "other", "split": split,
                        "scale": 1.4,  # 20% pad 等效
                        "sign_w": round(xyxy[pi][2] - xyxy[pi][0], 1),
                        "sign_h": round(xyxy[pi][3] - xyxy[pi][1], 1),
                        "source_image": f"{stem}.jpg", "origin": "detfp",
                        "src_class": "det", "det_conf": round(conf[pi], 4),
                    })
                    n_fp += 1
        print(f"detfp {split}: done ({time.time() - t0:.0f}s)", flush=True)
    return n_fp, n_img


def collect_detfp_test(rows: list[dict]) -> int:
    """test FPs from e2e_raw.json — eval-only negatives."""
    raw = json.loads(E2E_RAW.read_text(encoding="utf-8"))
    n = 0
    cache: dict[str, np.ndarray] = {}
    for rec in raw["records"]:
        for pr in rec["preds"]:
            if pr.get("matched"):
                continue
            img = cache.get(rec["image"])
            if img is None:
                p = find_image("test", rec["image"])
                if p is None:
                    continue
                img = cv2.imread(str(p))
                cache[rec["image"]] = img
                if img is None:
                    continue
            rel = f"images/detfp/test/{rec['image']}_p0.jpg"
            if crop_pad_save(img, pr["xyxy"], OUT / rel):
                rows.append({
                    "crop_relpath": rel, "old_name": "other", "split": "test",
                    "scale": 1.4,
                    "sign_w": round(pr["xyxy"][2] - pr["xyxy"][0], 1),
                    "sign_h": round(pr["xyxy"][3] - pr["xyxy"][1], 1),
                    "source_image": f"{rec['image']}.jpg", "origin": "detfp",
                    "src_class": "det", "det_conf": pr["det_conf"],
                })
                n += 1
    return n


def main() -> int:
    det_weights = ROOT / "runs" / "detector" / "smoke1" / "weights" / "best.pt"
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    print("1/3 label negs (pm/pr/ph) ...", flush=True)
    names = yaml.safe_load((ROOT / "configs" / "TT100K.yaml").read_text(encoding="utf-8"))["names"]
    if not isinstance(names, dict):
        raise SystemExit("TT100K.yaml names not a dict")
    counts = collect_label_negs({int(k): v for k, v in names.items()}, rows)
    print(f"   label negs: {counts}", flush=True)

    print("2/3 det fp on train/val (GPU, dir source) ...", flush=True)
    n_fp, n_img = collect_detfp_train_val(rows, det_weights)
    print(f"   detfp train/val: {n_fp} FPs (over {n_img} imgs with dets)", flush=True)

    print("3/3 det fp on test (from e2e_raw.json, eval-only) ...", flush=True)
    n_test = collect_detfp_test(rows)
    print(f"   detfp test: {n_test}", flush=True)

    with (OUT / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    from collections import Counter
    by_split = Counter(r["split"] for r in rows)
    print(f"total neg crops: {len(rows)}  by_split={dict(by_split)}")
    print(f"wrote {OUT / 'manifest.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
