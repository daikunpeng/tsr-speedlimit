#!/usr/bin/env python3
"""Self-domain detector score sweep: GT vs FP confidence distribution.

Motivation (2026-09-23, frame cam_1788765032_282571520_52):
  detector localizes the sign (IoU 0.90) but scores it 0.0049 — domain shift
  collapses confidence, and GT/FP score ranges overlap so threshold tuning
  cannot recover recall. Re-run after each annotation batch (prep_self_dataset
  --force) to track whether the distribution heals (after adaptation) or not.

Usage:
  .venv/bin/python scripts/self_score_sweep.py
  .venv/bin/python scripts/self_score_sweep.py --weights ... --scan-conf 0.0005

Outputs:
  reports/self_score_sweep.md   sweep table + per-GT-frame scores + quantiles
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = (0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002)


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


def load_boxes(path: Path, width: float, height: float) -> list[list[float]]:
    out = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) != 5:
                continue
            _, cx, cy, w, h = map(float, p)
            out.append([(cx - w / 2) * width, (cy - h / 2) * height,
                        (cx + w / 2) * width, (cy + h / 2) * height])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="runs/detector/smoke1/weights/best.pt")
    ap.add_argument("--data-root", default="data/self")
    ap.add_argument("--labels-sub", default="labels_det")
    ap.add_argument("--split", default="test")
    ap.add_argument("--src-size", default="640x352", metavar="WxH")
    ap.add_argument("--imgsz", type=int, nargs="+", default=[352, 640])
    ap.add_argument("--scan-conf", type=float, default=0.0005,
                    help="ultra-low floor so even collapsed GT scores are captured")
    ap.add_argument("--report", default="reports/self_score_sweep.md")
    ap.add_argument("--raw", default="runs/eval/self_score_sweep.json",
                    help="separate JSON path per model so comparisons do not overwrite the baseline")
    args = ap.parse_args()

    from ultralytics import YOLO

    width, height = (float(v) for v in args.src_size.lower().split("x"))
    data_root = ROOT / args.data_root
    img_dir = data_root / "images" / args.split
    lbl_dir = data_root / args.labels_sub / args.split
    imgsz = args.imgsz[0] if len(args.imgsz) == 1 else args.imgsz

    img_files = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not img_files:
        raise SystemExit(f"no images under {img_dir}")

    det = YOLO(args.weights)
    gt_rows: list[dict] = []   # per GT box: score (0 = no IoU>=0.5 pred at scan floor)
    fp_scores: list[float] = []
    n_img = len(img_files)

    for ip in img_files:
        key = ip.relative_to(img_dir).with_suffix("").as_posix()
        gts = load_boxes(lbl_dir / f"{key}.txt", width, height)
        r = next(det.predict(source=str(ip), imgsz=imgsz, conf=args.scan_conf,
                             iou=0.7, max_det=30, stream=True, verbose=False, device=0))
        preds = [([float(v) for v in b.xyxy[0]], float(b.conf)) for b in r.boxes]
        used: set[int] = set()
        for gi, g in enumerate(gts):
            best_iou, best_i = -1.0, None
            for i, (pb, _) in enumerate(preds):
                if i in used:
                    continue
                v = iou(g, pb)
                if v > best_iou:
                    best_iou, best_i = v, i
            if best_i is not None and best_iou >= 0.5:
                used.add(best_i)
                score = preds[best_i][1]
                gt_rows.append({"key": key, "gt": gi, "score": score, "iou": best_iou})
            else:
                gt_rows.append({"key": key, "gt": gi, "score": 0.0, "iou": 0.0})
        for i, (_, pc) in enumerate(preds):
            if i not in used:
                fp_scores.append(pc)

    gt_scores = np.array([r["score"] for r in gt_rows], dtype=float)
    fp_arr = np.array(fp_scores, dtype=float)
    n_gt = len(gt_scores)

    def qline(a: np.ndarray, qs=(0.0, 0.25, 0.5, 0.75, 1.0)) -> str:
        if a.size == 0:
            return "（空）"
        return "，".join(f"P{int(q * 100)}={float(np.quantile(a, q)):.5f}" for q in qs)

    lines = [
        "# Self 域 detector 分数扫描（GT vs FP conf 分布）",
        "",
        f"- 生成：{datetime.now(timezone.utc).isoformat()}；权重 `{args.weights}`",
        f"- 域：`{args.data_root}`（{args.labels_sub}）；{n_img} 图 / {n_gt} GT；"
        f"imgsz={imgsz}；扫描下限 conf≥{args.scan_conf}",
        "",
        "历史参考（smoke1）：帧 `cam_1788765032_282571520_52`（pl60，GT 短边 25.8px）"
        "定位 IoU 0.90 但 conf 仅 **0.0049**，GT crop 直接送 cls v2 → `pl60 0.992`；"
        "下文始终以本报告指定权重的实际分数为准。",
        "",
        "## 阈值扫描（IoU≥0.5 匹配）",
        "",
        "| conf 阈值 | recall | TP | FP 总数 | FP/img |",
        "|----------:|-------:|---:|--------:|-------:|",
    ]
    for th in THRESHOLDS:
        tp = int((gt_scores >= th).sum())
        fp = int((fp_arr >= th).sum())
        lines.append(f"| {th:g} | {tp / max(1, n_gt) * 100:.1f}% | {tp}/{n_gt} | {fp} | {fp / max(1, n_img):.3f} |")
    lines += [
        "",
        "## 分数分布（扫描下限处的原始 conf）",
        "",
        f"- **GT 匹配分数**：{qline(gt_scores)}",
        f"- **FP 分数**：{qline(fp_arr, (0.0, 0.5, 0.9, 0.99, 1.0))}",
        f"- GT 被彻底漏定位（无 IoU≥0.5 框）：{int((gt_scores == 0).sum())}/{n_gt}",
        "",
        "> 选阈值须同时考察 recall、FP 和逐事件稳定性；GT/FP 分数严重重叠时，"
        "仅调阈值不能同时改善召回与误报。",
        "",
        "## 逐 GT 框分数（升序）",
        "",
        "| score | IoU | 图 |",
        "|------:|----:|----|",
    ]
    for r in sorted(gt_rows, key=lambda r: r["score"]):
        lines.append(f"| {r['score']:.5f} | {r['iou']:.2f} | `{r['key']}` |")

    report = ROOT / args.report
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    raw = {"gt_rows": gt_rows, "fp_scores": fp_scores}
    raw_path = ROOT / args.raw
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(raw, indent=1), encoding="utf-8")
    print(f"GT recall@0.5={int((gt_scores >= 0.5).sum())}/{n_gt}, "
          f"median GT conf={float(np.median(gt_scores)):.5f}")
    print(f"wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
