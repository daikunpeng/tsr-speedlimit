#!/usr/bin/env python3
"""Per-size recall evaluation for the speed-limit detector on the official test split.

Coordinate-system convention (AGENTS.md):
  - GT labels and model predictions are matched in the ORIGINAL image frame
    (2048x2048 for TT100K); IoU is scale-invariant.
  - All SIZE buckets are reported in @640-EQUIVALENT pixels
    (px640 = px2048 * 0.3125), matching the deployment input resolution.
  - TT100K is square 2048x2048 -> scaling to 640x640 is a clean 0.3125 factor
    with NO letterbox padding. Non-square sources (GS130WI 640x352) must use
    the aspect-preserving scale factor instead — do not assume square.

Outputs (kept for analysis):
  runs/eval/<name>/size_eval_raw.json   per-GT match detail + per-image FP detail
  reports/detector_eval_size.md         bucketed recall report
  reports/figures/detector_size_recall.png
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DET_DIR = ROOT / "data" / "TT100K_speed_det"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

# bucket edges in @640-equivalent px (short side)
BUCKETS = [(0.0, 8.0, "<8"), (8.0, 12.0, "8-12"), (12.0, 16.0, "12-16"),
           (16.0, 24.0, "16-24"), (24.0, 32.0, "24-32"), (32.0, 48.0, "32-48"),
           (48.0, 1e9, ">=48")]
# deployment target range (user-confirmed): 12-28 px @640
TARGET_RANGE = (12.0, 28.0)
CONF_LEVELS = (0.10, 0.25, 0.50)
SRC_SIZE = 2048.0
TRAIN_SIZE = 640.0
SCALE_TO_640 = TRAIN_SIZE / SRC_SIZE  # 0.3125 (square TT100K, no letterbox)


def bucket_of(size640: float) -> str:
    for lo, hi, name in BUCKETS:
        if lo <= size640 < hi:
            return name
    return BUCKETS[-1][2]


def load_gt(split: str) -> dict[str, list[dict]]:
    """{image_name: [ {xyxy_2048, size640} ]}"""
    gts: dict[str, list[dict]] = {}
    lbl_dir = DET_DIR / "labels" / split
    for lf in sorted(lbl_dir.glob("*.txt")):
        boxes = []
        for line in lf.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) != 5:
                continue
            _, cx, cy, w, h = (float(x) for x in p)
            x1, y1 = (cx - w / 2) * SRC_SIZE, (cy - h / 2) * SRC_SIZE
            x2, y2 = (cx + w / 2) * SRC_SIZE, (cy + h / 2) * SRC_SIZE
            boxes.append({
                "xyxy": [x1, y1, x2, y2],
                "short_side_2048": min(x2 - x1, y2 - y1),
                "size640": min(x2 - x1, y2 - y1) * SCALE_TO_640,
            })
        gts[lf.stem] = boxes
    return gts


def iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """[N,4] and [M,4] xyxy -> [N,M] IoU."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))
    a = boxes1[:, None, :]
    b = boxes2[None, :, :]
    inter = np.maximum(0, np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0])) * \
        np.maximum(0, np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]))
    area1 = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area2 = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / (area1 + area2 - inter + 1e-9)


def main() -> int:
    from ultralytics import YOLO

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--match-iou", type=float, default=0.5)
    ap.add_argument("--name", default=None, help="eval output dir name (default: <split>_<weights_dir>)")
    args = ap.parse_args()

    wpath = Path(args.weights)
    name = args.name or f"{args.split}_{wpath.parent.parent.name}"
    out_dir = ROOT / "runs" / "eval" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    gts = load_gt(args.split)
    img_dir = DET_DIR / "images" / args.split
    img_files = sorted(
        p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    print(f"{args.split}: {len(img_files)} images, {sum(len(v) for v in gts.values())} GT boxes")

    model = YOLO(str(wpath))
    gt_records: list[dict] = []
    fp_records: list[dict] = []

    for img_path in img_files:
        stem = img_path.stem
        gt_boxes = gts.get(stem, [])
        preds = model.predict(str(img_path), imgsz=args.imgsz, conf=0.001,
                               iou=0.7, max_det=300, verbose=False, device=0)
        result = preds[0]
        if len(result.boxes) == 0:
            pred_xyxy = np.zeros((0, 4))
            pred_conf = np.zeros((0,))
        else:
            pred_xyxy = result.boxes.xyxy.cpu().numpy()
            pred_conf = result.boxes.conf.cpu().numpy()

        order = np.argsort(-pred_conf)
        matched_gt = set()
        n_gt = len(gt_boxes)
        for pi in order:  # greedy by conf
            best_j, best_iou = -1, args.match_iou
            for gj, gt in enumerate(gt_boxes):
                if gj in matched_gt:
                    continue
                ix = iou_matrix(pred_xyxy[pi:pi + 1], np.array([gt["xyxy"]]))[0, 0]
                if ix > best_iou:
                    best_j, best_iou = gj, ix
            if best_j >= 0:
                matched_gt.add(best_j)
                gt = gt_boxes[best_j]
                gt_records.append({
                    "image": stem, "gt_index": best_j,
                    "size640": round(gt["size640"], 2), "bucket": bucket_of(gt["size640"]),
                    "matched": True, "match_conf": round(float(pred_conf[pi]), 4),
                    "match_iou": round(float(best_iou), 4),
                })
            else:
                fp_records.append({
                    "image": stem, "xyxy": [round(float(v), 1) for v in pred_xyxy[pi]],
                    "conf": round(float(pred_conf[pi]), 4),
                })
        for gj, gt in enumerate(gt_boxes):
            if gj not in matched_gt:
                gt_records.append({
                    "image": stem, "gt_index": gj,
                    "size640": round(gt["size640"], 2), "bucket": bucket_of(gt["size640"]),
                    "matched": False,
                })

    # aggregate
    def recall(records, conf_min=None, bucket=None, size_range=None):
        sel = records
        if bucket is not None:
            sel = [r for r in sel if r["bucket"] == bucket]
        if size_range is not None:
            sel = [r for r in sel if size_range[0] <= r["size640"] < size_range[1]]
        if conf_min is not None:
            hit = [r for r in sel if r["matched"] and r["match_conf"] >= conf_min]
        else:
            hit = [r for r in sel if r["matched"]]
        return len(hit), len(sel), (len(hit) / len(sel) if sel else float("nan"))

    lines = [
        "# Detector 尺寸分桶评估（test split）",
        "",
        f"- 生成：{datetime.now(timezone.utc).isoformat()}；权重：`{wpath}`；imgsz={args.imgsz}；匹配 IoU≥{args.match_iou}",
        f"- 坐标系：匹配在原图 2048 空间；**尺寸桶为 @640 等效像素（×0.3125）**",
        "",
        "## 总体",
        "",
        "| conf 阈值 | recall | TP | GT |",
        "|-----------|-------:|---:|---:|",
    ]
    for c in CONF_LEVELS:
        tp, n, r = recall(gt_records, conf_min=c)
        lines.append(f"| ≥{c:.2f} | {r * 100:.1f}% | {tp} | {n} |")

    lines += ["", "## 尺寸分桶 recall", "",
              "| 桶（@640 px） | GT | recall@0.10 | recall@0.25 | recall@0.50 |",
              "|--------------|---:|------------:|------------:|------------:|"]
    for _, _, bname in BUCKETS:
        row = [bname]
        _, n, _ = recall(gt_records, bucket=bname)
        row.append(str(n))
        for c in CONF_LEVELS:
            _, _, r = recall(gt_records, conf_min=c, bucket=bname)
            row.append(f"{r * 100:.1f}%" if n else "—")
        lines.append("| " + " | ".join(row) + " |")
    _, n_t, _ = recall(gt_records, size_range=TARGET_RANGE)
    lines += ["", f"### 部署目标区间 [12, 28) px（GT {n_t} 个）", "",
              "| conf 阈值 | recall |", "|----------|-------:|"]
    for c in CONF_LEVELS:
        tp, n, r = recall(gt_records, conf_min=c, size_range=TARGET_RANGE)
        lines.append(f"| ≥{c:.2f} | {r * 100:.1f}% |")

    fp_img = defaultdict(int)
    for r in fp_records:
        fp_img[r["image"]] += 1
    n_imgs = len(img_files)
    for c in CONF_LEVELS:
        n_fp = sum(1 for r in fp_records if r["conf"] >= c)
        lines += [f"", f"- FP/img @conf≥{c:.2f}: {n_fp / n_imgs:.3f}（共 {n_fp} 个，含无 GT 负样本图）"]

    # save raw + report
    raw = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "weights": str(wpath), "split": args.split, "imgsz": args.imgsz,
        "match_iou": args.match_iou,
        "n_images": n_imgs, "n_gt": len(gt_records),
        "scale_to_640": SCALE_TO_640,
        "gt_records": gt_records, "fp_records": fp_records,
    }
    (out_dir / "size_eval_raw.json").write_text(
        json.dumps(raw, indent=1, ensure_ascii=False), encoding="utf-8")
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "detector_eval_size.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # figure: recall per bucket
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False
        FIGURES.mkdir(parents=True, exist_ok=True)
        names = [b[2] for b in BUCKETS]
        gtn = [recall(gt_records, bucket=b[2])[1] for b in BUCKETS]
        rs = {c: [recall(gt_records, conf_min=c, bucket=b[2])[2] * 100 for b in BUCKETS]
              for c in (0.10, 0.25)}
        x = np.arange(len(names))
        w = 0.38
        fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
        ax.bar(x - w / 2, rs[0.10], w, label="recall@conf≥0.10", color="#1f77b4")
        ax.bar(x + w / 2, rs[0.25], w, label="recall@conf≥0.25", color="#ff7f0e")
        for i, (v1, v2, n) in enumerate(zip(rs[0.10], rs[0.25], gtn)):
            ax.text(i - w / 2, v1 + 2, f"{v1:.0f}", ha="center", fontsize=8)
            ax.text(i + w / 2, v2 + 2, f"{v2:.0f}", ha="center", fontsize=8)
            ax.text(i, -14, f"GT {n}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_xlabel("sign 短边（@640 等效 px）")
        ax.set_ylabel("recall %")
        ax.set_ylim(0, 108)
        ax.set_title("Detector 尺寸→召回（test split）")
        ax.legend(loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(FIGURES / "detector_size_recall.png")
    except Exception as e:  # noqa: BLE001
        print(f"figure skipped: {e}")

    print(f"wrote {out_dir / 'size_eval_raw.json'}")
    print(f"wrote {REPORTS / 'detector_eval_size.md'}")
    tp, n, r = recall(gt_records, conf_min=0.25)
    print(f"overall recall@0.25: {r * 100:.1f}% ({tp}/{n})")
    _, n_t, _ = recall(gt_records, size_range=TARGET_RANGE)
    _, _, r_t = recall(gt_records, conf_min=0.25, size_range=TARGET_RANGE)
    print(f"target range 12-28px recall@0.25: {r_t * 100:.1f}% (n={n_t})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
