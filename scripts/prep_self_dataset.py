#!/usr/bin/env python3
"""Convert X-AnyLabeling JSON annotations under data/self/<trip>/ to TT100K-style layout.

Source (untouched):  data/self/<trip>/*.png|jpg  +  <stem>.json (missing json = negative frame)
Output:
  data/self/images/test/<trip>/<img>              real dir + per-file symlinks (原名, 可反解 trip)
  data/self/labels_det/test/<trip>/<stem>.txt     single-class 0 = speed_limit_sign (pl*/il* only)
  data/self/labels_mc/test/<trip>/<stem>.txt      speed class_map new_id (e2e classification GT)
  data/self/labels_lookalike/test/<trip>/<stem>.txt  classes.txt line idx (pm*/pr*/ph*/pne, FP attribution)
  data/self/meta/class_map.json                   copy of TT100K speed class_map
  data/self/manifest.csv                          per-frame inventory
  reports/self_dataset_stats.md                   distribution + suspects report

Rules:
  - shapes -> axis-aligned AABB (4-corner or 2-point rectangles only).
  - no-JSON frames get EMPTY txt in all three label dirs (TT100K negative convention).
  - lookalikes are deliberately NOT in labels_mc: eval_e2e greedy-matches mc boxes to
    det GT, a co-located pm box could pollute gt_cls.
  - fail-fast (no partial output): unknown label, leftover check_me, imagePath mismatch,
    JSON without image, non-rectangle shape.
  - suspects (slightly OOB / <1px / difficult) are KEPT and listed in the report
    (data rule: health-check only, never drop).
  - idempotent: existing outputs require --force to rebuild; source trips never modified.

Usage:
  .venv/bin/python scripts/prep_self_dataset.py            # first run
  .venv/bin/python scripts/prep_self_dataset.py --force    # always: full rebuild of ALL trips
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = ROOT / "data" / "self"
CLASSES = SELF / "classes.txt"
SPEED_MAP = ROOT / "data" / "TT100K_speed_meta" / "class_map.json"
REPORT = ROOT / "reports" / "self_dataset_stats.md"

OUT_DIRS = ("images", "labels_det", "labels_mc", "labels_lookalike", "meta")
IMG_SUFFIX = {".png", ".jpg", ".jpeg"}
BUCKETS = [(0, 8, "<8"), (8, 12, "8-12"), (12, 16, "12-16"), (16, 24, "16-24"),
           (24, 32, "24-32"), (32, 48, "32-48"), (48, 1e9, ">=48")]
TARGET = (12.0, 28.0)
OOB_EPS = 1.0  # px tolerance before flagging suspects


def bucket_of(s: float) -> str:
    for lo, hi, n in BUCKETS:
        if lo <= s < hi:
            return n
    return BUCKETS[-1][2]


def find_trips() -> list[Path]:
    trips = []
    for d in sorted(SELF.iterdir()):
        if not d.is_dir() or d.name in OUT_DIRS:
            continue
        if any(p.suffix.lower() in IMG_SUFFIX for p in d.iterdir()):
            trips.append(d)
    if not trips:
        raise SystemExit(f"no trip dirs with images under {SELF}")
    return trips


def rect_to_xyxy(shape: dict, img_w: int, img_h: int) -> tuple[list[float], list[str]]:
    """Return (xyxy pixels, suspects). Supports 4-corner or 2-point rectangles."""
    warns: list[str] = []
    st = shape.get("shape_type")
    if st != "rectangle":
        raise SystemExit(
            f"unsupported shape_type={st!r} label={shape.get('label')!r}; "
            f"only 'rectangle' is allowed (re-annotate polygons as boxes)")
    pts = shape.get("points") or []
    if len(pts) == 4:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    elif len(pts) == 2:
        (x1, y1), (x2, y2) = pts[0], pts[1]
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
    else:
        raise SystemExit(f"label={shape.get('label')!r}: points len {len(pts)} not in (2,4)")
    if x1 < -OOB_EPS or y1 < -OOB_EPS or x2 > img_w + OOB_EPS or y2 > img_h + OOB_EPS:
        warns.append(f"OOB ({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}) img={img_w}x{img_h}")
    if (x2 - x1) < 1 or (y2 - y1) < 1:
        warns.append(f"degenerate w={x2 - x1:.2f} h={y2 - y1:.2f}")
    if shape.get("difficult"):
        warns.append("difficult=true")
    return [x1, y1, x2, y2], warns


def yolo_line(cid: int, xyxy: list[float], img_w: int, img_h: int) -> str:
    x1, y1, x2, y2 = xyxy
    cx, cy = (x1 + x2) / 2 / img_w, (y1 + y2) / 2 / img_h
    w, h = (x2 - x1) / img_w, (y2 - y1) / img_h
    return f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="rebuild outputs for ALL trips (source JSON is the only truth)")
    args = ap.parse_args()

    classes = [ln.strip() for ln in CLASSES.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if "check_me" not in classes:
        raise SystemExit(f"{CLASSES}: missing check_me")
    cls_idx = {n: i for i, n in enumerate(classes)}
    speed_map = {r["old_name"]: r["new_id"] for r in json.loads(SPEED_MAP.read_text(encoding="utf-8"))}
    pl_il, look_pref = set(speed_map), ("pm", "pr", "ph", "pne")

    # outputs present?
    existing = [d for d in OUT_DIRS if (SELF / d).exists()]
    if existing and not args.force:
        raise SystemExit(f"outputs exist {existing}; rerun with --force to rebuild")
    if args.force:
        for d in OUT_DIRS:
            p = SELF / d
            if p.exists():
                shutil.rmtree(p)

    trips = find_trips()
    errors: list[str] = []
    suspects: list[str] = []
    manifest: list[dict] = []
    lab_counter: Counter[str] = Counter()
    trip_stats: dict[str, dict] = {}
    size_entries: list[tuple[str, float]] = []  # (class, size640_native)

    # ---- phase 1: validate everything (fail before any write) ----
    parsed: dict[str, dict] = {}  # trip -> stem -> record
    for trip in trips:
        recs: dict[str, dict] = {}
        imgs = sorted(p for p in trip.iterdir() if p.suffix.lower() in IMG_SUFFIX)
        n_json = n_neg = 0
        for img in imgs:
            stem = img.stem
            jp = trip / f"{stem}.json"
            shapes = []
            img_w, img_h = 640, 352  # GS130WI native; overridden by JSON header
            if jp.exists():
                n_json += 1
                d = json.loads(jp.read_text(encoding="utf-8"))
                ipath = d.get("imagePath")
                if ipath and Path(ipath).stem != stem:
                    errors.append(f"{trip.name}/{stem}: imagePath={ipath!r} mismatch")
                    continue
                iw, ih = d.get("imageWidth"), d.get("imageHeight")
                if (iw, ih) != (640, 352):
                    if not iw or not ih:
                        errors.append(f"{trip.name}/{stem}: missing imageWidth/Height")
                        continue
                    suspects.append(f"{trip.name}/{stem}: size {iw}x{ih} != 640x352")
                img_w, img_h = iw, ih
                for sh in d.get("shapes", []):
                    lab = sh.get("label", "")
                    if lab == "check_me":
                        errors.append(f"{trip.name}/{stem}: leftover check_me — "
                                      f"adjudicate in X-AnyLabeling first")
                    elif lab not in cls_idx:
                        errors.append(f"{trip.name}/{stem}: unknown label {lab!r} "
                                      f"(not in classes.txt)")
                        continue
                    xyxy, wlist = rect_to_xyxy(sh, img_w, img_h)
                    for w_ in wlist:
                        suspects.append(f"{trip.name}/{stem}:{lab} {w_}")
                    shapes.append({"label": lab, "xyxy": xyxy})
            else:
                n_neg += 1
            recs[stem] = {"img": img, "shapes": shapes, "has_json": jp.exists(),
                          "w": img_w, "h": img_h}
        # JSON without image
        for jp in trip.glob("*.json"):
            if not any((trip / f"{jp.stem}{s}").exists() for s in IMG_SUFFIX):
                errors.append(f"{trip.name}/{jp.name}: JSON without matching image")
        n_det = sum(1 for r in recs.values() for s in r["shapes"] if s["label"] in pl_il)
        n_look = sum(1 for r in recs.values() for s in r["shapes"]
                     if s["label"].startswith(look_pref))
        trip_stats[trip.name] = {"n_img": len(imgs), "n_json": n_json, "n_neg": n_neg,
                                 "n_det": n_det, "n_look": n_look}
        parsed[trip.name] = recs

    if errors:
        print("VALIDATION FAILED:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    # ---- phase 2: write ----
    for sub in ("images/test", "labels_det/test", "labels_mc/test", "labels_lookalike/test",
                "meta"):
        (SELF / sub).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SPEED_MAP, SELF / "meta" / "class_map.json")

    for trip_name, recs in parsed.items():
        (SELF / "images" / "test" / trip_name).mkdir(parents=True, exist_ok=True)
        for sub in ("labels_det", "labels_mc", "labels_lookalike"):
            (SELF / sub / "test" / trip_name).mkdir(parents=True, exist_ok=True)

        for stem, rec in sorted(recs.items()):
            img = rec["img"]
            dst = SELF / "images" / "test" / trip_name / img.name
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(img.resolve())

            det_l, mc_l, look_l = [], [], []
            for sh in rec["shapes"]:
                lab, xyxy = sh["label"], sh["xyxy"]
                lab_counter[lab] += 1
                if lab in pl_il:
                    det_l.append(yolo_line(0, xyxy, rec["w"], rec["h"]))
                    mc_l.append(yolo_line(speed_map[lab], xyxy, rec["w"], rec["h"]))
                    short_px = min(xyxy[2] - xyxy[0], xyxy[3] - xyxy[1])
                    size_entries.append((lab, short_px))  # native = @640 (factor 1.0)
                elif lab.startswith(look_pref):
                    look_l.append(yolo_line(cls_idx[lab], xyxy, rec["w"], rec["h"]))

            for sub, lines in (("labels_det", det_l), ("labels_mc", mc_l),
                               ("labels_lookalike", look_l)):
                p = SELF / sub / "test" / trip_name / f"{stem}.txt"
                p.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")

            manifest.append({
                "trip": trip_name, "image": img.name,
                "n_shape": len(rec["shapes"]), "n_det": sum(1 for s in rec["shapes"]
                                                            if s["label"] in pl_il),
                "n_lookalike": sum(1 for s in rec["shapes"]
                                   if s["label"].startswith(look_pref)),
                "has_json": int(rec["has_json"]), "width": rec["w"], "height": rec["h"],
            })

    with (SELF / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        wtr.writeheader()
        wtr.writerows(manifest)

    # ---- report ----
    n_img = sum(t["n_img"] for t in trip_stats.values())
    n_neg = sum(t["n_neg"] for t in trip_stats.values())
    n_det = sum(t["n_det"] for t in trip_stats.values())
    n_look = sum(t["n_look"] for t in trip_stats.values())
    bs: Counter[str] = Counter(bucket_of(s) for _, s in size_entries)
    t_n = sum(1 for _, s in size_entries if TARGET[0] <= s < TARGET[1])
    lines = [
        "# 自采数据集统计（data/self，纯测试集）",
        "",
        f"- 生成：{datetime.now(timezone.utc).isoformat()}；源格式：X-AnyLabeling JSON → TT100K YOLO txt",
        f"- 尺寸口径：**640×352 原生像素 = 部署输入 = @640 等效**（系数 1.0）；目标带 12–28px",
        "",
        "## 分 trip",
        "",
        "| trip | 图 | 有标注 | 负样本 | det 框(pl/il) | lookalike 框(pm/pr/ph/pne) |",
        "|------|---:|------:|------:|------------:|---------------------:|",
    ]
    for name, t in trip_stats.items():
        lines.append(f"| {name} | {t['n_img']} | {t['n_json']} | {t['n_neg']} | "
                     f"{t['n_det']} | {t['n_look']} |")
    lines += [
        f"| **合计** | **{n_img}** | {n_img - n_neg} | {n_neg} | **{n_det}** | {n_look} |",
        "",
        "## 类分布",
        "",
        "| label | count |", "|-------|------:|",
    ]
    for lab, c in lab_counter.most_common():
        lines.append(f"| {lab} | {c} |")
    lines += [
        "",
        f"## 尺寸分桶（det 框，原生 px，共 {len(size_entries)}）",
        "",
        "| 桶 | count |", "|----|------:|",
    ]
    for _, _, bn in BUCKETS:
        lines.append(f"| {bn} | {bs.get(bn, 0)} |")
    lines += [
        "",
        f"- **目标区间 [12,28)px：{t_n}/{len(size_entries)}"
        f"（{t_n / max(1, len(size_entries)) * 100:.0f}%）**",
        "",
        "## Suspects（保留，供人眼复核）",
        "",
    ]
    lines += [f"- {s}" for s in suspects] if suspects else ["- （无）"]
    lines += [
        "",
        "布局：`images/test/<trip>/`、`labels_det|mc|lookalike/test/<trip>/`；",
        "负样本三目录均为空 txt；`labels_mc` 仅 pl/il（防 pm 重叠框污染 gt_cls）；",
        "`pne`（禁止驶入，红圈）与 pl 同形 → lookalike 硬负样本。",
        "",
        f"清单：`{SELF / 'manifest.csv'}`",
        "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"trips {len(parsed)} | imgs {n_img} (neg {n_neg}) | det {n_det} | "
          f"lookalike {n_look} | suspects {len(suspects)}")
    print(f"wrote {SELF}/{{images,labels_det,labels_mc,labels_lookalike}}/test/ + manifest.csv")
    print(f"wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
