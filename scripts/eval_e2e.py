#!/usr/bin/env python3
"""End-to-end cascade evaluation: detector -> crop -> classifier.

Pipeline (user-confirmed 2026-09-23):
  test image (2048) -> YOLO detector (conf>=0.5) -> N boxes
    -> crop +20% pad -> resize 64x64 -> TinyCNN speed class
    -> greedy IoU>=0.5 match to GT; e2e TP = det TP AND correct class

Stability notes (2026-09-23 reboot lesson):
  - predict source MUST be a directory string (stream=True + wanted filter);
    a 3067-path list wedged the process (zero progress for 1h).
  - classifier runs BATCHED (256/batch) after detection, NOT per-crop GPU calls.

Outputs:
  runs/eval/<name>/e2e_raw.json       per-image detail
  reports/e2e_eval.md                  overall + size-bucketed report
  reports/figures/e2e_size_accuracy.png
  runs/vis/<name>/                     overlay images (optional --vis)
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DET = ROOT / "data" / "TT100K_speed_det"
MC = ROOT / "data" / "TT100K_speed_mc"
META = ROOT / "data" / "TT100K_speed_meta"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

BUCKETS = [(0.0, 8.0, "<8"), (8.0, 12.0, "8-12"), (12.0, 16.0, "12-16"),
           (16.0, 24.0, "16-24"), (24.0, 32.0, "24-32"), (32.0, 48.0, "32-48"),
           (48.0, 1e9, ">=48")]
TARGET = (12.0, 28.0)
SCALE_TO_640 = 640.0 / 2048.0
IOU = 0.5
PAD = 0.20
CLS_BS = 256


def bucket_of(s: float) -> str:
    for lo, hi, n in BUCKETS:
        if lo <= s < hi:
            return n
    return BUCKETS[-1][2]


def iou(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    s = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / s if s > 0 else 0.0


def load_boxes(path: Path, scale: float = 2048.0) -> list[dict]:
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
            "xyxy": [(cx - w / 2) * scale, (cy - h / 2) * scale,
                     (cx + w / 2) * scale, (cy + h / 2) * scale],
            "size640": min(w, h) * scale * SCALE_TO_640,
        })
    return out


def greedy_match(preds: list, gts: list, thr: float = IOU) -> tuple[dict[int, int], set[int]]:
    """Match preds (list order = priority) to gts by IoU.
    Returns {gt_idx: pred_idx}, set(matched pred idx)."""
    used: set[int] = set()
    match: dict[int, int] = {}
    for gi, g in enumerate(gts):
        best_pi, best_v = -1, thr
        for pi, p in enumerate(preds):
            if pi in used:
                continue
            v = iou(p["xyxy"], g["xyxy"])
            if v > best_v:
                best_pi, best_v = pi, v
        if best_pi >= 0:
            used.add(best_pi)
            match[gi] = best_pi
    return match, used


def crop_padded(img: np.ndarray, box) -> np.ndarray | None:
    """Crop with PAD ratio padding, resized to 64x64 uint8. None if degenerate."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * PAD, (y2 - y1) * PAD
    cx1, cy1 = max(0, int(x1 - px)), max(0, int(y1 - py))
    cx2, cy2 = min(w, int(x2 + px)), min(h, int(y2 + py))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    import cv2
    crop = cv2.resize(img[cy1:cy2, cx1:cx2], (64, 64), interpolation=cv2.INTER_LINEAR)
    # cv2.imread 是 BGR；classifier 训练用 PIL(RGB)。不转换会让蓝底 il 全被错分成 pl
    # （2026-09-23 实测 il 正确率 85% -> 10%）。可视化画框不走此函数，不受影响。
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def main() -> int:
    import cv2
    import torch
    from ultralytics import YOLO

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--det-weights", default="runs/detector/smoke1/weights/best.pt")
    ap.add_argument("--cls-weights", default="runs/classifier/smoke1/best.pt")
    ap.add_argument("--cls-map", default="runs/classifier/smoke1/class_map.json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--conf", type=float, default=0.50)
    ap.add_argument("--limit", type=int, default=0, help="0=all; uses dir source either way")
    ap.add_argument("--vis", action="store_true")
    ap.add_argument("--vis-limit", type=int, default=60)
    ap.add_argument("--name", default="e2e_test")
    args = ap.parse_args()

    out_dir = ROOT / "runs" / "eval" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = ROOT / "runs" / "vis" / args.name

    # ---- classifier ----
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location("tc", ROOT / "scripts" / "train_classifier.py")
    tc = module_from_spec(spec)
    spec.loader.exec_module(tc)
    cls_classes = json.loads(Path(args.cls_map).read_text(encoding="utf-8"))["classes"]
    id2cls = {v: k for k, v in cls_classes.items()}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cls_model = tc.TinyCNN(len(cls_classes)).to(dev)
    cls_model.load_state_dict(torch.load(args.cls_weights, map_location=dev, weights_only=True))
    cls_model.eval()

    cm = json.loads((META / "class_map.json").read_text(encoding="utf-8"))
    mc_id2name = {r["new_id"]: r["old_name"] for r in cm}

    # ---- GT ----
    det_gts = {lf.stem: load_boxes(lf) for lf in (DET / "labels" / args.split).glob("*.txt")}
    mc_gts = {lf.stem: load_boxes(lf) for lf in (MC / "labels" / args.split).glob("*.txt")}
    for stem, boxes in det_gts.items():
        mcs = mc_gts.get(stem, [])
        match, _ = greedy_match(mcs, boxes)  # {gt_idx=det_i: pred_idx=mc_i}
        for di, mi in match.items():
            boxes[di]["gt_cls"] = mc_id2name.get(mcs[mi]["cls_id"], f"cls{mcs[mi]['cls_id']}")

    img_dir = DET / "images" / args.split
    img_files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.limit:
        img_files = img_files[: args.limit]
    wanted = {p.stem for p in img_files}
    print(f"{args.split}: {len(wanted)} imgs, conf>={args.conf}, pad={PAD:.0%}, dev={dev}", flush=True)

    det = YOLO(args.det_weights)
    # 目录 source（stream 逐张产出；大 list source 曾致挂死，勿改回）
    results = det.predict(
        source=str(img_dir), imgsz=640, conf=args.conf, iou=0.7, max_det=100,
        stream=True, verbose=False, device=0,
    )

    # ---- Phase 1: detector stream -> match GT + collect crops ----
    records: list[dict] = []
    crop_refs: list[tuple[int, int]] = []   # (record_idx, pred_idx)
    crop_arrs: list[np.ndarray] = []        # uint8 64x64x3
    t0 = time.time()
    n_done = 0
    for r in results:
        stem = Path(r.path).stem
        if stem not in wanted:
            continue
        img = cv2.imread(r.path)
        if img is None:
            continue
        gts = det_gts.get(stem, [])
        if len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
        else:
            xyxy = np.zeros((0, 4), np.float32)
            conf = np.zeros((0,), np.float32)
        order = np.argsort(-conf)
        preds = []
        for pi in order:
            preds.append({
                "xyxy": [round(float(v), 1) for v in xyxy[pi]],
                "det_conf": round(float(conf[pi]), 4),
                "cls": "", "cls_conf": 0.0, "matched": False,
            })
            c = crop_padded(img, xyxy[pi])
            if c is None:
                c = np.zeros((64, 64, 3), np.uint8)
            crop_refs.append((len(records), len(preds) - 1))
            crop_arrs.append(c)

        g_match, matched_pi = greedy_match(preds, gts)
        gt_recs = []
        for gi, g in enumerate(gts):
            gr = {"size640": round(g["size640"], 2), "bucket": bucket_of(g["size640"]),
                  "det_tp": gi in g_match, "gt_cls": g.get("gt_cls")}
            if gi in g_match:
                matched_pi.add(g_match[gi])  # no-op, already there
                gr["pred_cls"] = ""  # filled after batch cls
            gt_recs.append(gr)
        # mark matched preds + link
        for gi, pi in g_match.items():
            preds[pi]["matched"] = True
            preds[pi]["gt_cls"] = gts[gi].get("gt_cls")

        records.append({
            "image": stem, "n_gt": len(gts), "n_det": len(preds),
            "det_tp": len(g_match), "det_fp": len(preds) - len(matched_pi),
            "det_fn": len(gts) - len(g_match),
            "preds": preds, "gt": gt_recs,
        })
        n_done += 1
        if n_done % 100 == 0:
            print(f"  det {n_done}/{len(wanted)} ({time.time() - t0:.0f}s)", flush=True)
        if args.limit and n_done >= len(wanted):
            break
    print(f"det done: {n_done} imgs, {len(crop_arrs)} crops ({time.time() - t0:.0f}s)", flush=True)

    # ---- Phase 2: batch classify ----
    if crop_arrs:
        batch = np.stack(crop_arrs)  # (N,64,64,3) uint8
        probs_all = np.zeros((len(batch), len(cls_classes)), np.float32)
        t1 = time.time()
        with torch.no_grad():
            for i in range(0, len(batch), CLS_BS):
                chunk = batch[i:i + CLS_BS]
                t = torch.from_numpy(chunk).permute(0, 3, 1, 2).float().to(dev) / 255.0
                t = (t - 0.5) / 0.5
                probs_all[i:i + len(chunk)] = cls_model(t).softmax(1).cpu().numpy()
        print(f"cls done: {len(batch)} crops in {time.time() - t1:.1f}s", flush=True)
        for (ri, pi), probs in zip(crop_refs, probs_all):
            idx = int(probs.argmax())
            rec = records[ri]
            rec["preds"][pi]["cls"] = id2cls[idx]
            rec["preds"][pi]["cls_conf"] = round(float(probs[idx]), 4)
            pr = rec["preds"][pi]
            if pr["matched"] and pr.get("gt_cls"):
                pr["cls_correct"] = (pr["cls"] == pr["gt_cls"])
        # fill gt_recs pred_cls/cls_correct for matched
        for rec in records:
            for gi, g in enumerate(rec["gt"]):
                if g["det_tp"]:
                    # find pred matched to this gt
                    for pi, pr in enumerate(rec["preds"]):
                        if pr["matched"] and pr.get("gt_cls") == g.get("gt_cls") and \
                           iou(pr["xyxy"], det_gts[rec["image"]][gi]["xyxy"]) >= IOU:
                            g["pred_cls"] = pr["cls"]
                            g["cls_correct"] = pr.get("cls_correct")
                            break

    # ---- aggregate ----
    n_img = len(records)
    total_gt = sum(r["n_gt"] for r in records)
    total_det = sum(r["n_det"] for r in records)
    total_tp = sum(r["det_tp"] for r in records)
    total_fp = sum(r["det_fp"] for r in records)
    total_fn = sum(r["det_fn"] for r in records)
    det_recall = total_tp / total_gt if total_gt else 0.0

    matched = [p for r in records for p in r["preds"] if p["matched"] and p.get("gt_cls")]
    cls_ok = sum(1 for p in matched if p.get("cls_correct"))
    cls_n = len(matched)
    cls_acc = cls_ok / cls_n if cls_n else 0.0
    e2e_acc = det_recall * cls_acc

    bs = defaultdict(lambda: {"gt": 0, "tp": 0, "cn": 0, "ok": 0})
    for r in records:
        for g in r["gt"]:
            b = bs[g["bucket"]]
            b["gt"] += 1
            if g["det_tp"]:
                b["tp"] += 1
                if g.get("gt_cls"):
                    b["cn"] += 1
                if g.get("cls_correct"):
                    b["ok"] += 1

    t_gt = t_tp = t_cn = t_ok = 0
    for r in records:
        for g in r["gt"]:
            if TARGET[0] <= g["size640"] < TARGET[1]:
                t_gt += 1
                if g["det_tp"]:
                    t_tp += 1
                    if g.get("gt_cls"):
                        t_cn += 1
                    if g.get("cls_correct"):
                        t_ok += 1
    t_dr = t_tp / t_gt if t_gt else 0
    t_ca = t_ok / t_cn if t_cn else 0

    fp_dist: dict[str, int] = defaultdict(int)
    for r in records:
        for p in r["preds"]:
            if not p["matched"]:
                fp_dist[p["cls"] or "None"] += 1

    # ---- report ----
    lines = [
        "# E2E 级联评估（detector → crop → classifier）",
        "",
        f"- 生成：{datetime.now(timezone.utc).isoformat()}；split={args.split}；"
        f"det conf≥{args.conf}；crop pad={PAD:.0%}；匹配 IoU≥{IOU}",
        f"- detector：`{args.det_weights}`",
        f"- classifier：`{args.cls_weights}`（{len(cls_classes)} 类，无 `other`）",
        f"- 图像 {n_img}；GT {total_gt}；检测框 {total_det}",
        "",
        "## 总体",
        "",
        "| 指标 | 值 |",
        "|------|----:|",
        f"| Det recall@{args.conf:.2f} | {det_recall * 100:.1f}%（{total_tp}/{total_gt}） |",
        f"| Det FP | {total_fp}（{total_fp / max(1, n_img):.3f}/img） |",
        f"| Det FN | {total_fn} |",
        f"| **Cls acc（det TP 且有 GT 类）** | **{cls_acc * 100:.1f}%**（{cls_ok}/{cls_n}） |",
        f"| **E2E acc = det recall × cls acc** | **{e2e_acc * 100:.1f}%** |",
        "",
        "## 尺寸分桶（@640 等效 px）",
        "",
        "| 桶 | GT | det recall | cls acc | e2e acc |",
        "|----|---:|----------:|--------:|--------:|",
    ]
    for _, _, bn in BUCKETS:
        b = bs[bn]
        if b["gt"] == 0:
            lines.append(f"| {bn} | 0 | — | — | — |")
            continue
        dr = b["tp"] / b["gt"]
        ca = b["ok"] / b["cn"] if b["cn"] else float("nan")
        ca_s = f"{ca * 100:.1f}%" if b["cn"] else "—"
        ea_s = f"{dr * ca * 100:.1f}%" if b["cn"] else "—"
        lines.append(f"| {bn} | {b['gt']} | {dr * 100:.1f}% | {ca_s} | {ea_s} |")
    lines += [
        "",
        f"### 部署目标区间 [12, 28) px（GT {t_gt}）",
        "",
        f"- det recall: **{t_dr * 100:.1f}%**（{t_tp}/{t_gt}）",
        f"- cls acc: **{t_ca * 100:.1f}%**（{t_ok}/{t_cn}）",
        f"- **e2e: {t_dr * t_ca * 100:.1f}%**",
        "",
        "## Detector FP → classifier 输出分布（`other` 类基线）",
        "",
        "（当前 classifier 无 `other` 类，FP 会被强制分到某个速度类）",
        "",
        "| cls | count |", "|-----|------:|",
    ]
    for cls, cnt in sorted(fp_dist.items(), key=lambda x: -x[1]):
        lines.append(f"| `{cls}` | {cnt} |")
    lines += ["", f"明细：`{out_dir / 'e2e_raw.json'}`", ""]
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "e2e_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    (out_dir / "e2e_raw.json").write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "det_weights": args.det_weights, "cls_weights": args.cls_weights,
        "split": args.split, "conf": args.conf, "pad": PAD, "iou_match": IOU,
        "n_images": n_img, "n_gt": total_gt, "n_det": total_det,
        "det_recall": round(det_recall, 4), "cls_acc": round(cls_acc, 4),
        "e2e_acc": round(e2e_acc, 4),
        "records": records,
    }, indent=1, ensure_ascii=False), encoding="utf-8")

    # ---- figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False
        FIGURES.mkdir(parents=True, exist_ok=True)
        names = [b[2] for b in BUCKETS]
        drs, cas, e2s = [], [], []
        for _, _, bn in BUCKETS:
            b = bs[bn]
            dr = b["tp"] / b["gt"] * 100 if b["gt"] else np.nan
            ca = b["ok"] / b["cn"] * 100 if b["cn"] else np.nan
            drs.append(dr)
            cas.append(ca)
            e2s.append(dr * ca / 100 if not np.isnan(dr) and not np.isnan(ca) else np.nan)
        x = np.arange(len(names))
        w = 0.27
        fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
        ax.bar(x - w, drs, w, label="det recall", color="#1f77b4")
        ax.bar(x, cas, w, label="cls acc", color="#2ca02c")
        ax.bar(x + w, e2s, w, label="e2e acc", color="#d62728")
        for i in range(len(names)):
            for vals, off in ((drs, -w), (cas, 0), (e2s, w)):
                if not np.isnan(vals[i]):
                    ax.text(i + off, vals[i] + 1, f"{vals[i]:.0f}", ha="center", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_xlabel("sign 短边（@640 等效 px）")
        ax.set_ylabel("%")
        ax.set_ylim(0, 115)
        ax.set_title(f"E2E 级联 conf≥{args.conf:g} pad={PAD:.0%}")
        ax.legend(loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(FIGURES / "e2e_size_accuracy.png")
    except Exception as e:  # noqa: BLE001
        print(f"figure skipped: {e}")

    # ---- vis ----
    if args.vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

        def badness(rec):
            cls_err = sum(1 for p in rec["preds"] if p["matched"] and p.get("gt_cls")
                          and not p.get("cls_correct"))
            return rec["det_fn"] * 3 + rec["det_fp"] * 2 + cls_err * 3
        picks = sorted(records, key=badness, reverse=True)[: args.vis_limit]
        n_vis = 0
        for rec in picks:
            stem = rec["image"]
            img_path = img_dir / f"{stem}.jpg"
            if not img_path.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            for g in det_gts.get(stem, []):
                x1, y1, x2, y2 = map(int, g["xyxy"])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 0), 2)
                if g.get("gt_cls"):
                    cv2.putText(img, g["gt_cls"], (x1, max(16, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 1, cv2.LINE_AA)
            for pr in rec["preds"]:
                x1, y1, x2, y2 = map(int, pr["xyxy"])
                if not pr["matched"]:
                    color = (255, 0, 255)      # FP magenta
                elif pr.get("cls_correct") is False:
                    color = (0, 165, 255)      # cls error orange
                else:
                    color = (0, 0, 255)        # TP red
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f"{pr['cls']} {pr['cls_conf']:.2f}",
                            (x1, min(img.shape[0] - 6, y2 + 16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            banner = (f"{stem} GT={rec['n_gt']} det={rec['n_det']} "
                      f"FN={rec['det_fn']} FP={rec['det_fp']}")
            cv2.rectangle(img, (0, 0), (img.shape[1], 28), (0, 0, 0), -1)
            cv2.putText(img, banner, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1, cv2.LINE_AA)
            scale = min(1.0, 1024 / max(img.shape[:2]))
            if scale < 1.0:
                img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
            cv2.imwrite(str(vis_dir / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            n_vis += 1
        rows = "".join(
            f'<div class=c><a href="{r["image"]}.jpg"><img src="{r["image"]}.jpg" loading=lazy></a>'
            f'<div class=b>{r["image"]} GT={r["n_gt"]} det={r["n_det"]} '
            f'FN={r["det_fn"]} FP={r["det_fp"]}</div></div>'
            for r in picks
        )
        (vis_dir / "index.html").write_text(
            f"""<!doctype html><meta charset=utf-8><title>{args.name}</title>
<style>body{{font-family:sans-serif;background:#111;color:#eee;margin:12px}}
.g{{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:10px}}
.c img{{width:100%;border:1px solid #444}}.b{{font-size:12px;padding:2px 0}}</style>
<h2>{args.name} conf≥{args.conf:g} — 绿框=GT / 红=TP / 橙=分类错 / 品红=FP</h2>
<div class=g>{rows}</div>""", encoding="utf-8")
        print(f"vis: {n_vis} -> {vis_dir}", flush=True)

    print(f"det recall@{args.conf:g}: {det_recall * 100:.1f}% ({total_tp}/{total_gt})")
    print(f"cls acc: {cls_acc * 100:.1f}% ({cls_ok}/{cls_n})")
    print(f"e2e acc: {e2e_acc * 100:.1f}%")
    print(f"target 12-28px: det {t_dr * 100:.1f}% cls {t_ca * 100:.1f}% e2e {t_dr * t_ca * 100:.1f}%")
    print(f"wrote {REPORTS / 'e2e_eval.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
