#!/usr/bin/env python3
"""Draw GT (green) and prediction (red) boxes on validation images for eyeballing.

Usage:
  .venv/bin/python scripts/vis_val_boxes.py \
      --weights runs/detector/smoke1/weights/best.pt \
      --split val [--limit N] [--conf 0.25]

Output: runs/vis/<name>/  (original-resolution JPGs with overlays)
Also writes an index.html with a simple grid for quick browsing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DET = ROOT / "data" / "TT100K_speed_det"


def draw_boxes(img: np.ndarray, boxes: np.ndarray, color: tuple[int, int, int], tag: str) -> None:
    for x1, y1, x2, y2 in boxes.astype(int):
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, tag, (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU for display")
    ap.add_argument("--limit", type=int, default=0, help="0 = all images")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    from ultralytics import YOLO

    wpath = Path(args.weights)
    name = args.name or f"{args.split}_{wpath.parent.parent.name}"
    out_dir = ROOT / "runs" / "vis" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    img_dir = DET / "images" / args.split
    lbl_dir = DET / "labels" / args.split
    img_files = sorted(
        p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if args.limit:
        img_files = img_files[: args.limit]
    print(f"{args.split}: {len(img_files)} images -> {out_dir}")

    model = YOLO(str(wpath))
    # batch predict for speed
    results = model.predict(
        str(img_dir), imgsz=args.imgsz, conf=args.conf, iou=args.iou,
        max_det=100, stream=True, verbose=False, device=0,
    )

    n_saved = 0
    n_tp = n_fp = n_fn = 0
    index_rows: list[str] = []
    wanted = {p.stem for p in img_files}
    for r in results:
        stem = Path(r.path).stem
        if stem not in wanted:
            continue

        img = cv2.imread(r.path)
        if img is None:
            continue
        h, w = img.shape[:2]

        # GT from label file (normalized xywh -> xyxy in original pixels)
        gt_boxes = []
        lf = lbl_dir / f"{stem}.txt"
        if lf.exists():
            for line in lf.read_text(encoding="utf-8").splitlines():
                p = line.split()
                if len(p) != 5:
                    continue
                _, cx, cy, bw, bh = (float(x) for x in p)
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
                gt_boxes.append([x1, y1, x2, y2])
        gt_arr = np.array(gt_boxes, dtype=np.float32)

        if len(r.boxes):
            pred_arr = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
        else:
            pred_arr = np.zeros((0, 4), dtype=np.float32)
            confs = np.zeros((0,), dtype=np.float32)

        # draw GT green, pred red (with conf)
        out = img.copy()
        draw_boxes(out, gt_arr, (0, 220, 0), "GT")
        for box, c in zip(pred_arr, confs):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(out, f"{c:.2f}", (x1, min(h - 4, y2 + 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

        # simple TP/FP/FN counts for the banner (IoU>=0.5 greedy)
        matched_pred = set()
        matched_gt = set()
        for gi, g in enumerate(gt_arr):
            for pi, p in enumerate(pred_arr):
                if pi in matched_pred:
                    continue
                ix1, iy1 = max(g[0], p[0]), max(g[1], p[1])
                ix2, iy2 = min(g[2], p[2]), min(g[3], p[3])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                a1 = (g[2] - g[0]) * (g[3] - g[1])
                a2 = (p[2] - p[0]) * (p[3] - p[1])
                if a1 + a2 - inter <= 0:
                    continue
                if inter / (a1 + a2 - inter) >= 0.5:
                    matched_pred.add(pi)
                    matched_gt.add(gi)
                    break
        tp = len(matched_gt)
        fp = len(pred_arr) - len(matched_pred)
        fn = len(gt_arr) - tp
        n_tp += tp
        n_fp += fp
        n_fn += fn

        banner = f"{stem}  GT={len(gt_arr)} P={len(pred_arr)}  TP={tp} FP={fp} FN={fn}"
        cv2.rectangle(out, (0, 0), (w, 26), (0, 0, 0), -1)
        cv2.putText(out, banner, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # downscale long side to 1024 for easier browsing (original kept? no — browsing copy)
        scale = min(1.0, 1024 / max(h, w))
        if scale < 1.0:
            out = cv2.resize(out, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        out_path = out_dir / f"{stem}.jpg"
        cv2.imwrite(str(out_path), out, [cv2.IMWRITE_JPEG_QUALITY, 90])
        n_saved += 1
        flag = "ok" if fn == 0 and fp == 0 else ("FN" if fn else "FP")
        index_rows.append(f'<div class="c"><a href="{out_path.name}"><img src="{out_path.name}" loading="lazy"></a><div class="b">{banner}</div></div>')
        if n_saved % 50 == 0:
            print(f"  saved {n_saved}")

    # index.html
    html = f"""<!doctype html><meta charset=utf-8><title>{name}</title>
<style>body{{font-family:sans-serif;background:#111;color:#eee;margin:12px}}
.g{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:10px}}
.c img{{width:100%;border:1px solid #444;background:#000}}
.b{{font-size:12px;padding:2px 0}}</style>
<h2>{name} — GT绿 / Pred红 conf≥{args.conf} — {n_saved} imgs, TP={n_tp} FP={n_fp} FN={n_fn}</h2>
<div class=g>{''.join(index_rows)}</div>
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    print(f"saved {n_saved} images -> {out_dir}")
    print(f"totals TP={n_tp} FP={n_fp} FN={n_fn}  (banner counts, IoU≥0.5)")
    print(f"open {out_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
