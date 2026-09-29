#!/usr/bin/env python3
"""Diagnose WHY large (>=24px) self signs are missed by the frozen detector.

Step A of the A->B plan (user 2026-09-24). Read-only on data. Joins each self GT
box (>=16px, focus >=24px) with the tt_crop1 detector scores and quantifies the
missed frames' appearance so the cause (top-edge clipping / dark or washed-out
render / motion-blur-or-defocus sharpness) can be eyeballed, not guessed.

Inputs:
  runs/eval/self_crop1/size_eval_raw.json   tt_crop1 det scores (conf floor 0.001)
  data/self/{labels_det}/test               GT boxes
  data/self/images/test                     images

Outputs:
  reports/diag_missed_frames.md
  runs/vis/missed_frames/                   contact sheet + per-frame JPGs

missed = no IoU>=0.5 box ("无定位") OR matched box conf < 0.5 ("定位低分").
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SELF_IMG = ROOT / "data" / "self" / "images" / "test"
SELF_DET = ROOT / "data" / "self" / "labels_det" / "test"
DET_EVAL = ROOT / "runs" / "eval" / "self_crop1" / "size_eval_raw.json"
FULL_W, FULL_H = 480, 264
ZOOM_SZ = 192


def load_det_boxes(key: str) -> list[dict]:
    p = SELF_DET / f"{key}.txt"
    out = []
    if p.exists():
        for ln in p.read_text(encoding="utf-8").splitlines():
            q = ln.split()
            if len(q) != 5:
                continue
            _, cx, cy, w, h = map(float, q)
            x1, y1 = (cx - w / 2) * 640, (cy - h / 2) * 352
            x2, y2 = (cx + w / 2) * 640, (cy + h / 2) * 352
            out.append({"xyxy": [x1, y1, x2, y2],
                        "size640": round(min(x2 - x1, y2 - y1), 2)})
    return out


def color_metrics(crop: np.ndarray) -> tuple[float, float]:
    h, w = crop.shape[:2]
    yy, xx = np.indices((h, w))
    q = ((xx + 0.5) / w - 0.5) ** 2 + ((yy + 0.5) / h - 0.5) ** 2
    ring = crop[(q >= 0.34 ** 2) & (q <= 0.50 ** 2)].astype(np.float32)
    core = crop[q < 0.25 ** 2]
    if len(ring) < 8 or len(core) < 8:
        return float("nan"), float("nan")
    red = float(np.median(ring[:, 2] - (ring[:, 0] + ring[:, 1]) / 2))
    luma = float(np.median(core[:, 0] * 0.114 + core[:, 1] * 0.587 + core[:, 2] * 0.299))
    return red, luma


def sharpness(crop_gray: np.ndarray) -> float:
    """Laplacian variance proxy: higher = sharper; small-sign caveat applies."""
    if crop_gray.size < 16:
        return float("nan")
    up = cv2.resize(crop_gray, (crop_gray.shape[1] * 4, crop_gray.shape[0] * 4),
                    interpolation=cv2.INTER_LINEAR)
    return float(cv2.Laplacian(up, cv2.CV_64F).var())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", default="reports/diag_missed_frames.md")
    ap.add_argument("--vis-dir", default="runs/vis/missed_frames")
    ap.add_argument("--min-size", type=float, default=16.0)
    ap.add_argument("--miss-size", type=float, default=24.0,
                    help=">= this size are 'large' misses worth eyeballing")
    args = ap.parse_args()

    ev = json.loads(DET_EVAL.read_text(encoding="utf-8"))
    by_img: dict[str, list[dict]] = defaultdict(list)
    for r in ev["gt_records"]:
        by_img[r["image"]].append(r)

    rows = []
    for key, recs in sorted(by_img.items()):
        img = cv2.imread(str(SELF_IMG / f"{key}.png"))
        if img is None:
            raise SystemExit(f"cannot read {key}")
        boxes = load_det_boxes(key)
        for r in recs:
            if r["size640"] < args.min_size:
                continue
            b = boxes[r["gt_index"]]
            x1, y1, x2, y2 = b["xyxy"]
            crop = img[max(0, int(y1)):min(352, int(y2)) + 1,
                       max(0, int(x1)):min(640, int(x2)) + 1]
            red, luma = color_metrics(crop)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sharp = sharpness(gray[max(0, int(y1)):min(352, int(y2)) + 1,
                                   max(0, int(x1)):min(640, int(x2)) + 1])
            conf = r["match_conf"] if r.get("matched") else 0.0
            rows.append({
                "key": key, "frame": int(key.rsplit("_", 1)[1]),
                "trip": key.split("/", 1)[0],
                "size640": round(r["size640"], 1),
                "y1": round(y1, 1), "y_norm": round((y1 + y2) / 2 / 352, 2),
                "clipped": y1 < 3.0,
                "red": round(red, 1), "luma": round(luma, 1),
                "sharp": round(sharp, 1),
                "conf": round(conf, 4),
                "status": "检出" if conf >= 0.5 else ("定位低分" if r.get("matched") else "无定位"),
                "box": b["xyxy"],
            })

    rows.sort(key=lambda r: (-r["size640"], r["key"]))
    missed = [r for r in rows if r["status"] != "检出"]
    miss_big = [r for r in missed if r["size640"] >= args.miss_size]
    det_big = [r for r in rows if r["status"] == "检出" and r["size640"] >= args.miss_size]

    def med(a):
        a = [v for v in a if v == v]
        return round(float(np.median(a)), 1) if a else float("nan")

    lines = [
        "# 漏检帧成因诊断（tt_crop1 冻结检测器，self 开发集）",
        "",
        f"- 生成：{datetime.now(timezone.utc).isoformat()}；检测分数来自 `{DET_EVAL}`（conf 下限 0.001）",
        f"- 口径：检出=conf≥0.5；定位低分=有 IoU≥0.5 框但 conf<0.5；无定位=连框都没有",
        "- sharp=Laplacian 方差（×4 上采样后），越高越锐；小牌上噪声大，仅作相对参考",
        "",
        f"## 概览（≥{args.min_size:g}px）",
        "",
        f"- 合计 {len(rows)}；检出 {sum(r['status']=='检出' for r in rows)}；"
        f"定位低分 {sum(r['status']=='定位低分' for r in rows)}；"
        f"无定位 {sum(r['status']=='无定位' for r in rows)}",
        f"- **大牌漏检（≥{args.miss_size:g}px）{len(miss_big)} 个**；大牌检出 {len(det_big)} 个",
        "",
        "## 大牌：检出 vs 漏检 指标对比",
        "",
        "| 组 | n | 尺寸P50 | y1 中位 | 上缘裁切 | 红度P50 | 亮度P50 | sharpP50 |",
        "|----|---:|--------:|--------:|--------:|--------:|--------:|---------:|",
    ]
    for name, sel in (("检出(conf≥0.5)", det_big), ("漏检(无定位+低分)", miss_big)):
        if not sel:
            lines.append(f"| {name} | 0 | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {name} | {len(sel)} | {med([r['size640'] for r in sel]):.0f} | "
            f"{med([r['y1'] for r in sel]):.0f} | {sum(r['clipped'] for r in sel)}/{len(sel)} | "
            f"{med([r['red'] for r in sel]):.0f} | {med([r['luma'] for r in sel]):.0f} | "
            f"{med([r['sharp'] for r in sel]):.0f} |")
    lines += ["", f"## 大牌漏检逐帧（≥{args.miss_size:g}px，按尺寸降序）", "",
              "| trip | 帧 | 尺寸 | y1 | 裁切? | 红度 | 亮度 | sharp | conf | 状态 |",
              "|------|---:|-----:|---:|:---:|-----:|-----:|------:|------:|------|"]
    for r in miss_big:
        lines.append(f"| {r['trip']} | {r['frame']} | {r['size640']:.0f} | {r['y1']:.0f} | "
                     f"{'是' if r['clipped'] else ''} | {r['red']:.0f} | {r['luma']:.0f} | "
                     f"{r['sharp']:.0f} | {r['conf']:.4f} | {r['status']} |")

    report = ROOT / args.report
    report.write_text("\n".join(lines), encoding="utf-8")

    # ---- contact sheet of large misses ----
    vis = ROOT / args.vis_dir
    vis.mkdir(parents=True, exist_ok=True)
    for r in miss_big:
        key = r["key"]
        img = cv2.imread(str(SELF_IMG / f"{key}.png"))
        x1, y1, x2, y2 = map(int, r["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 0), 2)
        banner = (f"{r['frame']} {r['size640']:.0f}px y1={r['y1']:.0f}{' 裁切' if r['clipped'] else ''} "
                  f"red={r['red']:.0f} lum={r['luma']:.0f} sharp={r['sharp']:.0f} conf={r['conf']:.4f}")
        bar = np.full((26, img.shape[1], 3), 20, np.uint8)
        cv2.putText(bar, banner, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        full = cv2.resize(np.vstack([bar, img]), (FULL_W, FULL_H + 26), interpolation=cv2.INTER_AREA)
        px, py = (x2 - x1) * 0.3, (y2 - y1) * 0.3
        cx1, cy1 = max(0, int(x1 - px)), max(0, int(y1 - py))
        cx2, cy2 = min(640, int(x2 + px)), min(352, int(y2 + py))
        zc = cv2.resize(img[cy1:cy2, cx1:cx2], (ZOOM_SZ, ZOOM_SZ), interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(zc, (0, 0), (ZOOM_SZ - 1, ZOOM_SZ - 1), (0, 220, 0), 2)
        zcanvas = np.full((FULL_H + 26, ZOOM_SZ, 3), 12, np.uint8)
        off = (FULL_H + 26 - ZOOM_SZ) // 2
        zcanvas[off:off + ZOOM_SZ, :] = zc
        fname = vis / f"{r['trip']}_{r['frame']}_{r['size640']:.0f}px.jpg"
        cv2.imwrite(str(fname), np.hstack([full, zcanvas]), [cv2.IMWRITE_JPEG_QUALITY, 92])

    print(f"rows {len(rows)}; missed(≥16px) {len(missed)}; big misses {len(miss_big)}")
    print(f"wrote {report}; vis -> {vis}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
