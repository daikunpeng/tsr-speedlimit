#!/usr/bin/env python3
"""Frozen-weight diagnostics on data/self: color, position, GT-crop classification.

No training, no data mutation. Answers three questions with the FROZEN smoke1
detector + cls v2 (see reports/detector_smear_ablation.md for why pixel smear
was not the bottleneck):

  1. Does cls v2 read self digits from GT crops? (font/plate-spec factor)
  2. Are self plate colors (ring redness, center luma) far from TT100K 24-48px
     references, and do they correlate with detector scores?
  3. Does vertical position (top-of-frame share) explain detection failures
     beyond size?

Inputs (all pre-existing):
  runs/eval/self_torch/size_eval_raw.json      smoke1 det scores on self (conf floor 0.001)
  runs/eval/smoke1_test/size_eval_raw.json     smoke1 det scores on TT100K test
  data/self/{labels_det,labels_mc}/test        GT boxes + speed classes
  data/TT100K_speed_mc/labels/test             TT100K speed classes (reference colors)
  runs/classifier/v2/{best.pt,class_map.json}  frozen classifier

Outputs:
  reports/frozen_diag.md
  runs/eval/frozen_diag/raw.json

Notes:
  - cv2 reads BGR; cls v2 trains on PIL RGB. Crops are converted explicitly
    (2026-09-23 lesson: missing this conversion mis-classified all il plates).
  - Ring/core metrics are heuristics for diagnosis only; they never gate data.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SELF = ROOT / "data" / "self"
TT_MC = ROOT / "data" / "TT100K_speed_mc" / "labels" / "test"
TT_IMG = ROOT / "data" / "TT100K_speed_det" / "images" / "test"
TT_DET_EVAL = ROOT / "runs" / "eval" / "smoke1_test" / "size_eval_raw.json"
SELF_DET_EVAL = ROOT / "runs" / "eval" / "self_torch" / "size_eval_raw.json"
SPEED_MAP = ROOT / "data" / "TT100K_speed_meta" / "class_map.json"
CLS_W = ROOT / "runs" / "classifier" / "v2" / "best.pt"
CLS_MAP = ROOT / "runs" / "classifier" / "v2" / "class_map.json"
PAD = 0.20
CLS_BS = 256
TARGET_CLASS = (24.0, 48.0)


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_boxes(path: Path, width: float, height: float) -> list[dict]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        cid, cx, cy, w, h = int(float(p[0])), float(p[1]), float(p[2]), float(p[3]), float(p[4])
        out.append({
            "cls_id": cid,
            "xyxy": [(cx - w / 2) * width, (cy - h / 2) * height,
                     (cx + w / 2) * width, (cy + h / 2) * height],
        })
    return out


def gt_class_for(det_box: list[float], mc_boxes: list[dict]) -> str | None:
    best, best_i = 0.5, -1
    for i, m in enumerate(mc_boxes):
        v = iou(det_box, m["xyxy"])
        if v > best:
            best, best_i = v, i
    return None if best_i < 0 else f"mc{mc_boxes[best_i]['cls_id']}"


def crop_padded(img: np.ndarray, box) -> np.ndarray | None:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * PAD, (y2 - y1) * PAD
    cx1, cy1 = max(0, int(x1 - px)), max(0, int(y1 - py))
    cx2, cy2 = min(w, int(x2 + px)), min(h, int(y2 + py))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    crop = cv2.resize(img[cy1:cy2, cx1:cx2], (64, 64), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)  # cv2=BGR, cls trains on RGB


def color_metrics(crop: np.ndarray) -> tuple[float, float]:
    """Return (ring red excess, core luma) medians; annulus excludes corners."""
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cls-weights", default=str(CLS_W),
                    help="classifier checkpoint (default: frozen cls v2)")
    ap.add_argument("--cls-map", default=str(CLS_MAP),
                    help="classifier class map json (default: cls v2)")
    ap.add_argument("--report", default="reports/frozen_diag.md")
    ap.add_argument("--raw-out", default="runs/eval/frozen_diag/raw.json",
                    help="separate raw path per classifier so runs do not overwrite")
    args = ap.parse_args()

    import torch

    cls_w = ROOT / args.cls_weights
    cls_map_path = ROOT / args.cls_map
    spec = spec_from_file_location("tc", ROOT / "scripts" / "train_classifier.py")
    tc = module_from_spec(spec)
    spec.loader.exec_module(tc)
    cls_classes = json.loads(cls_map_path.read_text(encoding="utf-8"))["classes"]
    id2cls = {v: k for k, v in cls_classes.items()}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = tc.TinyCNN(len(cls_classes)).to(dev)
    model.load_state_dict(torch.load(cls_w, map_location=dev, weights_only=True))
    model.eval()

    def cls_probs(batch: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.from_numpy(batch).permute(0, 3, 1, 2).float().to(dev) / 255.0
            t = (t - 0.5) / 0.5
            return model(t).softmax(1).cpu().numpy()

    mc_id2name = {r["new_id"]: r["old_name"] for r in json.loads(SPEED_MAP.read_text(encoding="utf-8"))}

    # ---------- self: per-GT-box det score + color + GT-crop cls ----------
    self_eval = json.loads(SELF_DET_EVAL.read_text(encoding="utf-8"))
    by_img: dict[str, list[dict]] = defaultdict(list)
    for r in self_eval["gt_records"]:
        by_img[r["image"]].append(r)

    rows: list[dict] = []
    crop_buf: list[np.ndarray] = []
    for key, recs in sorted(by_img.items()):
        trip, stem = key.split("/", 1)
        img = cv2.imread(str(SELF / "images" / "test" / f"{key}.png"))
        if img is None:
            raise SystemExit(f"cannot read self image {key}")
        det_boxes = load_boxes(SELF / "labels_det" / "test" / f"{key}.txt", 640, 352)
        mc_boxes = load_boxes(SELF / "labels_mc" / "test" / f"{key}.txt", 640, 352)
        for r in recs:
            box = det_boxes[r["gt_index"]]["xyxy"]
            best, best_i = 0.5, -1
            for i, m in enumerate(mc_boxes):
                v = iou(box, m["xyxy"])
                if v > best:
                    best, best_i = v, i
            gt_cls = mc_id2name.get(mc_boxes[best_i]["cls_id"]) if best_i >= 0 else None
            red, luma = color_metrics(img[int(box[1]):int(box[3]) + 1, int(box[0]):int(box[2]) + 1])
            crop = crop_padded(img, box)
            if crop is None:
                crop = np.zeros((64, 64, 3), np.uint8)
            crop_buf.append(crop)
            rows.append({
                "trip": trip, "image": key, "size640": r["size640"], "bucket": r["bucket"],
                "y_norm": round((box[1] + box[3]) / 2 / 352, 3),
                "det_conf": round(r["match_conf"], 4) if r.get("matched") else 0.0,
                "det_iou": r.get("match_iou", 0.0),
                "gt_cls": gt_cls, "ring_red": round(red, 1), "core_luma": round(luma, 1),
            })

    probs_all = np.zeros((len(crop_buf), len(cls_classes)), np.float32)
    for i in range(0, len(crop_buf), CLS_BS):
        probs_all[i:i + CLS_BS] = cls_probs(np.stack(crop_buf[i:i + CLS_BS]))
    for row, probs in zip(rows, probs_all):
        idx = int(probs.argmax())
        row["crop_pred"] = id2cls[idx]
        row["crop_conf"] = round(float(probs[idx]), 4)
        row["crop_correct"] = bool(row["gt_cls"]) and row["crop_pred"] == row["gt_cls"]

    # ---------- TT100K test reference: color of 24-48px pl20/30/40/60 ----------
    tt_eval = json.loads(TT_DET_EVAL.read_text(encoding="utf-8"))
    want = {"pl20", "pl30", "pl40", "pl60"}
    tt_by_img: dict[str, list[dict]] = defaultdict(list)
    for r in tt_eval["gt_records"]:
        if TARGET_CLASS[0] <= r["size640"] < TARGET_CLASS[1]:
            tt_by_img[r["image"]].append(r)
    tt_rows: list[dict] = []
    for key, recs in sorted(tt_by_img.items()):
        img = cv2.imread(str(TT_IMG / f"{key}.jpg"))
        if img is None:
            continue
        mc_boxes = load_boxes(TT_MC / f"{key}.txt", 2048, 2048)
        det_boxes = load_boxes(ROOT / "data" / "TT100K_speed_det" / "labels" / "test" / f"{key}.txt", 2048, 2048)
        for r in recs:
            box = det_boxes[r["gt_index"]]["xyxy"]
            best, best_i = 0.5, -1
            for i, m in enumerate(mc_boxes):
                v = iou(box, m["xyxy"])
                if v > best:
                    best, best_i = v, i
            name = mc_id2name.get(mc_boxes[best_i]["cls_id"]) if best_i >= 0 else None
            if name not in want:
                continue
            # @640-equivalent crop for color (scale 0.3125, INTER_AREA like deployment)
            x1, y1, x2, y2 = box
            cw, ch = max(4, round((x2 - x1) * 0.3125)), max(4, round((y2 - y1) * 0.3125))
            crop = cv2.resize(img[int(y1):int(y2) + 1, int(x1):int(x2) + 1], (cw, ch),
                              interpolation=cv2.INTER_AREA)
            red, luma = color_metrics(crop)
            tt_rows.append({
                "image": key, "cls": name, "size640": r["size640"],
                "y_norm": round((box[1] + box[3]) / 2 / 2048, 3),
                "det_conf": round(r["match_conf"], 4) if r.get("matched") else 0.0,
                "ring_red": round(red, 1), "core_luma": round(luma, 1),
            })

    # ---------- aggregate ----------
    def med(a):
        a = [v for v in a if v == v]
        return float(np.median(a)) if a else float("nan")

    lines = [
        "# 冻结权重诊断：颜色 / 位置 / GT-crop 分类（self 开发集）",
        "",
        f"- 生成：{datetime.now(timezone.utc).isoformat()}；detector=smoke1（分数来自既有评估，conf 下限 0.001），"
        f"classifier=`{args.cls_weights}`（GT crop、20% pad、64×64、BGR→RGB）",
        "- 颜色指标为**诊断用启发式**（红圈 R−(G+B)/2 中位、核心亮度中位），不作数据筛选依据。",
        "- det_conf=0 表示无 IoU≥0.5 匹配框；y_norm 为框中心纵向归一化位置（0=顶）。",
        "",
        "## 1. GT-crop 分类（font/牌面规格因素的直接度量）",
        "",
    ]
    n_all = len(rows)
    n_ok = sum(r["crop_correct"] for r in rows)
    has_cls = [r for r in rows if r["gt_cls"]]
    n_ok_c = sum(r["crop_correct"] for r in has_cls)
    lines += [
        f"- 全部 GT 框 {n_all}（有速度类 {len(has_cls)}）：crop 分类正确 **{n_ok_c}/{len(has_cls)}"
        f"（{n_ok_c / max(1, len(has_cls)) * 100:.1f}%）**；被判 `other`："
        f"{sum(r['crop_pred'] == 'other' for r in has_cls)}",
        "",
        "| 尺寸桶 | n | crop 分类正确 | →other |",
        "|--------|---:|--------------:|-------:|",
    ]
    buckets = ["<8", "8-12", "12-16", "16-24", "24-32", "32-48", ">=48"]
    for b in buckets:
        sel = [r for r in has_cls if r["bucket"] == b]
        lines.append(f"| {b} | {len(sel)} | {sum(r['crop_correct'] for r in sel)} | "
                     f"{sum(r['crop_pred'] == 'other' for r in sel)} |")
    lines += ["", "### 分类错误明细（GT crop，含低置信）", "",
              "| trip | 图 | GT | 预测 | conf | 尺寸 |", "|------|----|----|------|-----:|-----:|"]
    for r in has_cls:
        if not r["crop_correct"]:
            lines.append(f"| {r['trip']} | `{r['image']}` | {r['gt_cls']} | {r['crop_pred']} | "
                         f"{r['crop_conf']:.2f} | {r['size640']:.0f}px |")
    if all(r["crop_correct"] for r in has_cls):
        lines.append("| （无） | | | | | |")

    lines += ["", "## 2. 逐 trip 颜色/位置/检测概况（全部 GT 框）", "",
              "| trip | n | det conf P50 | 无定位框 | y_norm<0.2 占比 | 红圈红度 P50 | 核心亮度 P50 | crop 分类正确 |",
              "|------|---:|------------:|---------:|----------------:|-------------:|-------------:|--------------:|"]
    trips = sorted({r["trip"] for r in rows})
    for t in trips:
        sel = [r for r in rows if r["trip"] == t]
        cls_sel = [r for r in sel if r["gt_cls"]]
        lines.append(
            f"| {t} | {len(sel)} | {med([r['det_conf'] for r in sel]):.4f} | "
            f"{sum(r['det_conf'] == 0 for r in sel)} | "
            f"{sum(r['y_norm'] < 0.2 for r in sel) / len(sel) * 100:.0f}% | "
            f"{med([r['ring_red'] for r in sel]):.0f} | {med([r['core_luma'] for r in sel]):.0f} | "
            f"{sum(r['crop_correct'] for r in cls_sel)}/{len(cls_sel)} |")

    lines += ["", f"## 3. 目标带 24–48px：self vs TT100K test（pl20/30/40/60）", "",
              "| 域 | n | det conf P50 | y_norm<0.2 | 红圈红度 P50 | 核心亮度 P50 |",
              "|----|---:|------------:|------------:|-------------:|-------------:|"]
    sel24 = [r for r in rows if TARGET_CLASS[0] <= r["size640"] < TARGET_CLASS[1]]
    for dom, sel in (("self(全部牌)", sel24), ("TT100K(pl20/30/40/60)", tt_rows)):
        if not sel:
            lines.append(f"| {dom} | 0 | — | — | — | — |")
            continue
        lines.append(f"| {dom} | {len(sel)} | {med([r['det_conf'] for r in sel]):.4f} | "
                     f"{sum(r['y_norm'] < 0.2 for r in sel) / len(sel) * 100:.0f}% | "
                     f"{med([r['ring_red'] for r in sel]):.0f} | {med([r['core_luma'] for r in sel]):.0f} |")
    tt_by_cls = defaultdict(list)
    for r in tt_rows:
        tt_by_cls[r["cls"]].append(r)
    lines += ["", "TT100K 分类别参考：", ""]
    for c in sorted(tt_by_cls):
        sel = tt_by_cls[c]
        lines.append(f"- {c}: n={len(sel)}，红度 P50 {med([r['ring_red'] for r in sel]):.0f}，"
                     f"亮度 P50 {med([r['core_luma'] for r in sel]):.0f}，det conf P50 "
                     f"{med([r['det_conf'] for r in sel]):.3f}")

    lines += ["", "### self 24–48px 逐框（按 det conf 升序）", "",
              "| trip | GT | 尺寸 | y_norm | 红度 | 亮度 | det conf | crop 预测 |",
              "|------|----|-----:|-------:|-----:|-----:|---------:|----------|"]
    for r in sorted(sel24, key=lambda r: r["det_conf"]):
        lines.append(f"| {r['trip']} | {r['gt_cls'] or '—'} | {r['size640']:.0f} | {r['y_norm']:.2f} | "
                     f"{r['ring_red']:.0f} | {r['core_luma']:.0f} | {r['det_conf']:.4f} | "
                     f"{r['crop_pred']}{'✓' if r['crop_correct'] else ''} |")

    top = [r for r in sel24 if r["y_norm"] < 0.2]
    mid = [r for r in sel24 if r["y_norm"] >= 0.2]
    red_hi = [r for r in sel24 if r["ring_red"] >= 28]  # TT pl40 P50 reference
    red_lo = [r for r in sel24 if r["ring_red"] < 28]
    lines += [
        "",
        "## 4. 简单相关性（24–48px，self）",
        "",
        f"- 上缘（y_norm<0.2）n={len(top)}：det conf P50 {med([r['det_conf'] for r in top]):.4f}；"
        f"中部/下部 n={len(mid)}：P50 {med([r['det_conf'] for r in mid]):.4f}",
        f"- 红度 ≥ TT pl40 参考值(28) n={len(red_hi)}：det conf P50 "
        f"{med([r['det_conf'] for r in red_hi]):.4f}；<28 n={len(red_lo)}："
        f"{med([r['det_conf'] for r in red_lo]):.4f}",
        f"- crop 分类正确率（24–48px 有速度类）："
        f"{sum(r['crop_correct'] for r in sel24 if r['gt_cls'])}/{sum(1 for r in sel24 if r['gt_cls'])}",
        "",
        "> 样本量小（数十框），以上只用于指出方向，不构成因果结论；",
        "> 位置与颜色在现有 6 个 trip 内高度共变，需由 TT100K 侧的受控增广实验分离。",
        "",
    ]

    report = ROOT / args.report
    report.write_text("\n".join(lines), encoding="utf-8")
    raw_path = ROOT / args.raw_out
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps({"self_rows": rows, "tt_rows": tt_rows},
                                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"self rows {len(rows)} (24-48px: {len(sel24)}), tt rows {len(tt_rows)}")
    print(f"crop cls acc {n_ok_c}/{len(has_cls)}; wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
