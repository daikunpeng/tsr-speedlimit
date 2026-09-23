#!/usr/bin/env python3
"""Generate sign-centric crops from TT100K speed-limit multiclass subset.

Sources: data/TT100K_speed_mc (46-class YOLO labels) + data/TT100K_speed_meta/class_map.json
Output:  data/TT100K_speed_crops/ (images + manifest.csv + meta/crop_stats.json)
Report:  reports/crop_stats.md (size / class health check for the tiny classifier)

Scales (strategy.md §3): tight(1.0) / 1.3 / 1.5 / 2.0 context around each box.
Crops keep original pixels (no resize / upscale); degradation is a later augmentation step.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC_MC = ROOT / "data" / "TT100K_speed_mc"
CLASS_MAP_JSON = ROOT / "data" / "TT100K_speed_meta" / "class_map.json"
OUT_DIR = ROOT / "data" / "TT100K_speed_crops"
META_DIR = OUT_DIR / "meta"
REPORTS = ROOT / "reports"
SPLITS = ("train", "val", "test")
DEFAULT_SCALES = (1.0, 1.3, 1.5, 2.0)
JPEG_QUALITY = 95
DATASET_SOURCE = "TT100K"

# sign short-side buckets (px) — foundation of the core KPI: size -> accuracy
SIZE_BUCKETS = (8, 12, 16, 24, 32, 64)

MANIFEST_FIELDS = [
    "crop_relpath",
    "split",
    "source_image",
    "ann_index",
    "scale",
    "new_id",
    "old_name",
    "group",
    "speed_value",
    "bbox_norm_cx",
    "bbox_norm_cy",
    "bbox_norm_w",
    "bbox_norm_h",
    "bbox_x1",
    "bbox_y1",
    "sign_w",
    "sign_h",
    "crop_w",
    "crop_h",
    "source_w",
    "source_h",
    "dataset_source",
    "camera_id",
]


def load_class_map() -> list[dict]:
    rows = json.loads(CLASS_MAP_JSON.read_text(encoding="utf-8"))
    if not rows:
        raise SystemExit(f"empty {CLASS_MAP_JSON}")
    return rows


def scale_tag(scale: float) -> str:
    return f"s{int(round(scale * 10)):02d}"


def expand_box(x1: float, y1: float, w: float, h: float, scale: float,
               img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """Expand box around its center by `scale`, clamp to image bounds."""
    if scale <= 1.0:
        cx1, cy1, cw, ch = x1, y1, w, h
    else:
        cx, cy = x1 + w / 2.0, y1 + h / 2.0
        cw, ch = w * scale, h * scale
        cx1, cy1 = cx - cw / 2.0, cy - ch / 2.0
    ix1 = max(0, min(int(math.floor(cx1)), img_w - 1))
    iy1 = max(0, min(int(math.floor(cy1)), img_h - 1))
    ix2 = max(0, min(int(math.ceil(cx1 + cw)), img_w))
    iy2 = max(0, min(int(math.ceil(cy1 + ch)), img_h))
    return ix1, iy1, ix2 - ix1, iy2 - iy1


def parse_label_file(path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cid, cx, cy, w, h = (float(parts[0]), *map(float, parts[1:]))
        boxes.append((int(cid), cx, cy, w, h))
    return boxes


def find_image(split: str, stem: str) -> Path | None:
    img_dir = SRC_MC / "images" / split
    for ext in (".jpg", ".jpeg", ".png"):
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    i = min(len(sorted_vals) - 1, max(0, int(round(q / 100.0 * (len(sorted_vals) - 1)))))
    return sorted_vals[i]


def dry_run(scales: list[float]) -> int:
    n_boxes = 0
    per_split: dict[str, int] = {}
    short_sides: list[float] = []
    for split in SPLITS:
        cnt = 0
        lbl_dir = SRC_MC / "labels" / split
        if not lbl_dir.is_dir():
            print(f"missing {lbl_dir}", file=sys.stderr)
            return 1
        for lf in lbl_dir.glob("*.txt"):
            for cid, _, _, w, h in parse_label_file(lf):
                cnt += 1
                # nominal 2048x2048; real size read lazily, header only
                short_sides.append(min(w, h) * 2048)
        per_split[split] = cnt
        n_boxes += cnt
    total_crops = n_boxes * len(scales)
    ss = sorted(short_sides)
    print(f"boxes: {per_split}  total={n_boxes}")
    print(f"scales: {scales} -> estimated crops = {total_crops}")
    print(f"sign short-side px (nominal 2048): P10={percentile(ss,10):.0f} "
          f"P50={percentile(ss,50):.0f} P90={percentile(ss,90):.0f}")
    print("output would be written to", OUT_DIR)
    return 0


def run(scales: list[float], force: bool) -> int:
    if OUT_DIR.exists() and not force:
        if (META_DIR / "crop_stats.json").exists():
            print("crops exist; use --force to rebuild", file=sys.stderr)
            return 1
    if force and OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for split in SPLITS:
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    class_rows = load_class_map()
    by_new_id = {r["new_id"]: r for r in class_rows}

    manifest_path = OUT_DIR / "manifest.csv"
    n_saved = 0
    n_skipped = 0
    split_stats: list[dict] = []
    class_counter: Counter[str] = Counter()
    split_class_counter: dict[str, Counter[str]] = defaultdict(Counter)
    short_sides: list[float] = []
    sizes_by_class: dict[str, list[float]] = defaultdict(list)

    with manifest_path.open("w", newline="", encoding="utf-8") as mf:
        writer = csv.DictWriter(mf, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for split in SPLITS:
            lbl_dir = SRC_MC / "labels" / split
            files = sorted(lbl_dir.glob("*.txt"))
            n_split = 0
            n_split_skipped = 0
            for lf in tqdm(files, desc=split, unit="img"):
                stem = lf.stem
                img_path = find_image(split, stem)
                if img_path is None:
                    n_skipped += 1
                    n_split_skipped += 1
                    continue
                with Image.open(img_path) as im:
                    W, H = im.size
                    boxes = parse_label_file(lf)
                    for i, (cid, ncx, ncy, nw, nh) in enumerate(boxes):
                        row = by_new_id.get(cid)
                        if row is None:
                            n_skipped += 1
                            n_split_skipped += 1
                            continue
                        x1, y1, w, h = (ncx - nw / 2) * W, (ncy - nh / 2) * H, nw * W, nh * H
                        if w < 1 or h < 1:
                            n_skipped += 1
                            n_split_skipped += 1
                            continue
                        short_sides.append(min(w, h))
                        sizes_by_class[row["old_name"]].append(min(w, h))
                        for scale in scales:
                            bx, by, bw, bh = expand_box(x1, y1, w, h, scale, W, H)
                            if bw < 2 or bh < 2:
                                n_skipped += 1
                                n_split_skipped += 1
                                continue
                            rel = f"images/{split}/{stem}_i{i:02d}_{scale_tag(scale)}.jpg"
                            out_p = OUT_DIR / rel
                            if not out_p.exists():
                                im.crop((bx, by, bx + bw, by + bh)).save(
                                    out_p, quality=JPEG_QUALITY
                                )
                            writer.writerow({
                                "crop_relpath": rel,
                                "split": split,
                                "source_image": img_path.name,
                                "ann_index": i,
                                "scale": scale,
                                "new_id": cid,
                                "old_name": row["old_name"],
                                "group": row["group"],
                                "speed_value": row["speed_value"],
                                "bbox_norm_cx": f"{ncx:.6f}",
                                "bbox_norm_cy": f"{ncy:.6f}",
                                "bbox_norm_w": f"{nw:.6f}",
                                "bbox_norm_h": f"{nh:.6f}",
                                "bbox_x1": int(bx),
                                "bbox_y1": int(by),
                                "sign_w": round(w, 1),
                                "sign_h": round(h, 1),
                                "crop_w": bw,
                                "crop_h": bh,
                                "source_w": W,
                                "source_h": H,
                                "dataset_source": DATASET_SOURCE,
                                "camera_id": "",
                            })
                            n_saved += 1
                            n_split += 1
                            class_counter[row["old_name"]] += 1
                            split_class_counter[split][row["old_name"]] += 1
            split_stats.append({
                "split": split,
                "n_boxes": sum(split_class_counter[split][c] // len(scales)
                               for c in split_class_counter[split]),
                "n_crops": n_split,
                "n_skipped": n_split_skipped,
            })

    stats = build_stats(scales, n_saved, n_skipped, split_stats,
                        class_counter, split_class_counter, short_sides,
                        sizes_by_class, class_rows)
    (META_DIR / "crop_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(stats, REPORTS / "crop_stats.md")
    print(f"wrote {manifest_path}")
    print(f"wrote {META_DIR / 'crop_stats.json'}")
    print(f"wrote {REPORTS / 'crop_stats.md'}")
    for s in split_stats:
        print(f"  {s['split']}: crops={s['n_crops']} skipped={s['n_skipped']}")
    return 0


def build_stats(scales, n_saved, n_skipped, split_stats, class_counter,
                split_class_counter, short_sides, sizes_by_class, class_rows):
    ss = sorted(short_sides)
    buckets = defaultdict(int)
    for v in ss:
        placed = False
        for b in SIZE_BUCKETS:
            if v < b:
                buckets[f"<{b}"] += 1
                placed = True
                break
        if not placed:
            buckets[f">={SIZE_BUCKETS[-1]}"] += 1
    name_of = {r["old_name"]: r for r in class_rows}
    class_table = []
    for name, cnt in sorted(class_counter.items(), key=lambda kv: -kv[1]):
        cvals = sorted(sizes_by_class[name])
        class_table.append({
            "old_name": name,
            "group": name_of[name]["group"],
            "speed_value": name_of[name]["speed_value"],
            "n_boxes": cnt // len(scales),
            "n_crops": cnt,
            "p50_short_side_px": round(percentile(cvals, 50), 1),
            "share_lt_16px": round(
                sum(1 for v in cvals if v < 16) / max(1, len(cvals)) * 100, 1),
        })
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(SRC_MC),
        "out": str(OUT_DIR),
        "scales": scales,
        "n_signs": len(ss),
        "n_crops": n_saved,
        "n_skipped": n_skipped,
        "splits": split_stats,
        "short_side_px": {
            "p5": round(percentile(ss, 5), 1),
            "p10": round(percentile(ss, 10), 1),
            "p25": round(percentile(ss, 25), 1),
            "p50": round(percentile(ss, 50), 1),
            "p75": round(percentile(ss, 75), 1),
            "p90": round(percentile(ss, 90), 1),
            "min": round(ss[0], 1) if ss else None,
            "max": round(ss[-1], 1) if ss else None,
        },
        "size_buckets": dict(buckets),
        "classes": class_table,
    }


def write_report(stats: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ss = stats["short_side_px"]
    lines = [
        "# Sign crop 数据体检（TT100K 限速子集）",
        "",
        f"生成时间：{stats['generated_at']}；来源：`data/TT100K_speed_mc`；"
        f"scales：{stats['scales']}（crop 保持原始像素，不做放大）",
        "",
        "## 总量",
        "",
        "| split | signs | crops | skipped |",
        "|-------|-------|-------|---------|",
    ]
    for s in stats["splits"]:
        lines.append(f"| {s['split']} | {s['n_boxes']} | {s['n_crops']} | {s['n_skipped']} |")
    lines += [
        "",
        f"合计：signs **{stats['n_signs']}**，crops **{stats['n_crops']}**，skipped {stats['n_skipped']}",
        "",
        "## 尺寸分布（sign 短边，px）——核心 KPI 基础",
        "",
        f"P5 {ss['p5']} / P10 {ss['p10']} / P25 {ss['p25']} / P50 {ss['p50']} / "
        f"P75 {ss['p75']} / P90 {ss['p90']} / min {ss['min']} / max {ss['max']}",
        "",
        "| 桶（短边 px） | 数量 | 占比 |",
        "|--------------|------|------|",
    ]
    total = max(1, stats["n_signs"])
    order = [f"<{b}" for b in SIZE_BUCKETS] + [f">={SIZE_BUCKETS[-1]}"]
    for k in order:
        v = stats["size_buckets"].get(k, 0)
        lines.append(f"| {k} | {v} | {v / total * 100:.1f}% |")
    lines += [
        "",
        "## 类别分布（长尾检查）",
        "",
        "| 类 | 组 | 速度值 | signs | crops | P50 短边 px | <16px 占比 |",
        "|----|----|--------|-------|-------|-----------|------------|",
    ]
    for c in stats["classes"]:
        lines.append(
            f"| {c['old_name']} | {c['group']} | {c['speed_value']} | {c['n_boxes']} "
            f"| {c['n_crops']} | {c['p50_short_side_px']} | {c['share_lt_16px']}% |"
        )
    lines += [
        "",
        "## 风险提示",
        "",
        f"- val split 的 sign 只有 **{stats['splits'][1]['n_boxes']}** 个，"
        "分类验证集偏小：评估分类器时建议 train/test 互补或后续自采数据补充。",
        "- 小于 16px 的 sign 占比高（见尺寸分布），正对应「远距离能检出、易分错」——"
        "分类训练需按 strategy.md 做 small-object degradation 增强。",
        "- 长尾类（signs < 50）建议合并到速度值层面或后续补数据，训练时用 domain-balanced sampling。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Count boxes / estimate crops without writing")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild output directory (delete existing crops)")
    parser.add_argument("--scales", default=",".join(str(s) for s in DEFAULT_SCALES),
                        help="Comma-separated box expansion scales (default 1.0,1.3,1.5,2.0)")
    args = parser.parse_args()

    scales = sorted({float(x) for x in args.scales.split(",")})
    if not scales or any(s <= 0 for s in scales):
        print("invalid scales", file=sys.stderr)
        return 1

    if not (SRC_MC / "labels").is_dir():
        print(f"missing {SRC_MC / 'labels'} — run scripts/filter_speed_limit.py first",
              file=sys.stderr)
        return 1
    if not CLASS_MAP_JSON.exists():
        print(f"missing {CLASS_MAP_JSON}", file=sys.stderr)
        return 1

    if args.dry_run:
        return dry_run(scales)
    return run(scales, args.force)


if __name__ == "__main__":
    raise SystemExit(main())
