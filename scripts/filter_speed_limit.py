#!/usr/bin/env python3
"""Filter TT100K into speed subsets (pl*+il*) and a weight-limit archive (pm*).

Class semantics (verified 2026-09-23, see reports/label_audit.md):
  pl* = max speed limit (white bg, red ring, black digits)      -> speed positive
  il* = minimum speed limit (BLUE bg, white digits, 50-110)     -> speed positive
  pm* = weight limit (white bg, red ring, black "Nt", tonnes)   -> NOT speed;
        archived separately as hard-negative reserve
Earlier project docs had pm/il meanings swapped; data itself was fine.

Outputs:
  data/TT100K_speed_mc/  + configs/TT100K_speed.yaml      (pl+il, renumbered 0..N)
  data/TT100K_speed_det/ + configs/TT100K_speed_det.yaml  (single class speed_limit_sign)
  data/TT100K_weight_pm/  + configs/TT100K_weight_pm.yaml  (pm archive, not a training subset)
  data/TT100K_speed_meta/{class_map.json, weight_class_map.json, filter_stats.json}
  reports/speed_filter_stats.md

Exclusions: --exclude <json> drops individual source boxes, e.g. confirmed
pl<->il mislabels from reports/label_audit.md. Format:
  [{"split": "train", "image": "60423.jpg", "source_line": 3, "reason": "..."}]
where source_line is the 0-based line index in the original TT100K label file.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SRC_DIR = DATA_DIR / "TT100K"
SRC_YAML = ROOT / "configs" / "TT100K.yaml"
OUT_MC = DATA_DIR / "TT100K_speed_mc"
OUT_DET = DATA_DIR / "TT100K_speed_det"
OUT_WEIGHT = DATA_DIR / "TT100K_weight_pm"
META_DIR = DATA_DIR / "TT100K_speed_meta"
CONFIGS = ROOT / "configs"
REPORTS = ROOT / "reports"
SPLITS = ("train", "val", "test")

SPEED_PREFIXES = ("pl", "il")   # positives: max speed + minimum speed
WEIGHT_PREFIXES = ("pm",)        # weight limit: archived, not speed
VALUE_RE = re.compile(r"^(?:pl|pm|il)(\d+(?:\.\d+)?)$", re.IGNORECASE)


def load_names(path: Path) -> dict[int, str]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = raw["names"]
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(n) for i, n in enumerate(names)}


def is_speed_class(name: str) -> bool:
    return any(name.startswith(p) for p in SPEED_PREFIXES)


def is_weight_class(name: str) -> bool:
    return any(name.startswith(p) for p in WEIGHT_PREFIXES)


def semantic_of(name: str) -> str:
    if name.startswith("pl"):
        return "speed_max"
    if name.startswith("il"):
        return "speed_min"
    if name.startswith("pm"):
        return "weight"
    return "other"


def parse_value(name: str) -> float | None:
    m = VALUE_RE.match(name)
    return float(m.group(1)) if m else None


def build_class_map(names: dict[int, str], keep: set[str]) -> list[dict]:
    """Rows for classes whose prefix is in `keep`, sorted by name, renumbered from 0."""
    rows_src = [(oid, n) for oid, n in names.items() if n.startswith(tuple(keep))]
    rows_src.sort(key=lambda x: x[1])
    rows = []
    for new_id, (oid, name) in enumerate(rows_src):
        rows.append(
            {
                "old_id": oid,
                "old_name": name,
                "new_id": new_id,
                "group": name[:2],
                "semantic": semantic_of(name),
                "speed_value": parse_value(name),
            }
        )
    return rows


def ensure_images_link(out_root: Path, force: bool) -> None:
    link = out_root / "images"
    target = SRC_DIR / "images"
    if link.is_symlink() or link.exists():
        if force:
            if link.is_symlink() or link.is_file():
                link.unlink()
            elif link.is_dir():
                shutil.rmtree(link)
        else:
            return
    link.symlink_to(target.resolve(), target_is_directory=True)


def prepare_out_root(out_root: Path, force: bool) -> None:
    if out_root.exists() and force:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    ensure_images_link(out_root, force=force)


def write_yaml(path: Path, out_root: Path, class_map: list[dict], header: str) -> None:
    doc = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {row["new_id"]: row["old_name"] for row in class_map},
    }
    path.write_text(
        header + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def percentiles(values: list[float], ps: tuple[float, ...] = (50, 75, 90, 95)) -> dict[str, float]:
    if not values:
        return {f"p{int(p)}": 0.0 for p in ps}
    xs = sorted(values)
    out: dict[str, float] = {}
    for p in ps:
        if len(xs) == 1:
            out[f"p{int(p)}"] = xs[0]
            continue
        k = (len(xs) - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            out[f"p{int(p)}"] = xs[int(k)]
        else:
            out[f"p{int(p)}"] = xs[f] * (c - k) + xs[c] * (k - f)
    return out


def load_exclusions(path: Path | None) -> set[tuple[str, str, int]]:
    """Return {(split, label_filename, source_line_idx)}."""
    if path is None:
        return set()
    rows = json.loads(path.read_text(encoding="utf-8"))
    keys = set()
    for r in rows:
        img = r["image"]
        name = img if img.endswith(".txt") else img.rsplit(".", 1)[0] + ".txt"
        keys.add((r["split"], name, int(r["source_line"])))
    print(f"loaded {len(keys)} exclusions from {path}")
    return keys


def process_split(
    split: str,
    old_to_new_speed: dict[int, int],
    speed_old_ids: set[int],
    old_to_new_weight: dict[int, int],
    weight_old_ids: set[int],
    exclude_keys: set[tuple[str, str, int]],
) -> dict:
    src_label_dir = SRC_DIR / "labels" / split
    src_image_dir = SRC_DIR / "images" / split
    mc_label_dir = OUT_MC / "labels" / split
    det_label_dir = OUT_DET / "labels" / split
    weight_label_dir = OUT_WEIGHT / "labels" / split

    n_src_labels = 0
    n_out_labels = 0
    n_with_speed = 0
    n_empty_neg = 0
    n_speed_kept = 0
    n_weight_kept = 0
    n_boxes_dropped = 0
    n_boxes_excluded = 0
    per_class: Counter = Counter()
    weight_per_class: Counter = Counter()
    widths: list[float] = []
    heights: list[float] = []

    img_files = [
        p
        for p in src_image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    n_images = len(img_files)

    for label_path in sorted(src_label_dir.glob("*.txt")):
        n_src_labels += 1
        mc_lines: list[str] = []
        det_lines: list[str] = []
        weight_lines: list[str] = []
        kept_boxes: list[tuple[int, float, float, float, float]] = []

        for line_idx, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            oid = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:5])
            if (split, label_path.name, line_idx) in exclude_keys:
                n_boxes_excluded += 1
                continue
            if oid in speed_old_ids:
                new_id = old_to_new_speed[oid]
                mc_lines.append(f"{new_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                det_lines.append(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                kept_boxes.append((oid, x, y, w, h))
                per_class[oid] += 1
                n_speed_kept += 1
            elif oid in weight_old_ids:
                weight_lines.append(
                    f"{old_to_new_weight[oid]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                )
                weight_per_class[oid] += 1
                n_weight_kept += 1
            else:
                n_boxes_dropped += 1

        # Speed subsets: always write label file if source had one (empty = negative)
        (mc_label_dir / label_path.name).write_text(
            ("\n".join(mc_lines) + ("\n" if mc_lines else "")), encoding="utf-8"
        )
        (det_label_dir / label_path.name).write_text(
            ("\n".join(det_lines) + ("\n" if det_lines else "")), encoding="utf-8"
        )
        n_out_labels += 1
        # Weight archive: only files that contain weight boxes
        if weight_lines:
            (weight_label_dir / label_path.name).write_text(
                "\n".join(weight_lines) + "\n", encoding="utf-8"
            )
        if kept_boxes:
            n_with_speed += 1
            img_path = src_image_dir / f"{label_path.stem}.jpg"
            if not img_path.exists():
                for ext in (".jpeg", ".png", ".JPG"):
                    cand = src_image_dir / f"{label_path.stem}{ext}"
                    if cand.exists():
                        img_path = cand
                        break
            if img_path.exists():
                try:
                    with Image.open(img_path) as im:
                        iw, ih = im.size
                    for _, _, _, w, h in kept_boxes:
                        widths.append(w * iw)
                        heights.append(h * ih)
                except OSError:
                    pass
        else:
            n_empty_neg += 1

    return {
        "split": split,
        "n_images": n_images,
        "n_src_label_files": n_src_labels,
        "n_out_label_files": n_out_labels,
        "n_images_with_speed": n_with_speed,
        "n_empty_negatives": n_empty_neg,
        "n_speed_boxes": n_speed_kept,
        "n_weight_boxes": n_weight_kept,
        "n_boxes_excluded": n_boxes_excluded,
        "n_boxes_dropped": n_boxes_dropped,
        "per_class_old_id": dict(per_class),
        "weight_per_class_old_id": dict(weight_per_class),
        "bbox_width_px": percentiles(widths),
        "bbox_height_px": percentiles(heights),
        "n_bbox_sized": len(widths),
    }


def write_report(
    speed_map: list[dict],
    weight_map: list[dict],
    split_stats: list[dict],
    stats_path: Path,
    md_path: Path,
    exclude_used: bool,
) -> None:
    name_by_old = {r["old_id"]: r["old_name"] for r in speed_map + weight_map}
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(SRC_DIR),
        "semantics": {
            "pl": "max speed limit (speed positive)",
            "il": "minimum speed limit (speed positive)",
            "pm": "weight limit in tonnes (NOT speed; archived)",
        },
        "out_mc": str(OUT_MC),
        "out_det": str(OUT_DET),
        "out_weight": str(OUT_WEIGHT),
        "n_speed_classes": len(speed_map),
        "n_weight_classes": len(weight_map),
        "exclusions_applied": exclude_used,
        "speed_class_map": speed_map,
        "weight_class_map": weight_map,
        "splits": split_stats,
    }
    stats_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# TT100K speed-limit filter stats（语义修正版）",
        "",
        f"- Generated: `{payload['generated_at']}`",
        f"- Source: `{SRC_DIR}`",
        f"- Speed multiclass out: `{OUT_MC}`（pl*+il*，{len(speed_map)} 类）",
        f"- Detector out: `{OUT_DET}`（单类 speed_limit_sign）",
        f"- Weight archive out: `{OUT_WEIGHT}`（pm*，{len(weight_map)} 类，**非速度，不参与训练**）",
        "- 语义：pl=最高限速、il=最低限速（蓝底）、pm=限重（吨）；"
        "plx/pmx/ilx 无实例，未纳入（见 reports/label_audit.md）",
        "",
        "## Speed classes",
        "",
        "| new_id | old_id | name | group | semantic | value |",
        "|-------:|-------:|------|-------|----------|------:|",
    ]
    for r in speed_map:
        sv = r["speed_value"]
        sv_s = "" if sv is None else str(sv)
        lines.append(
            f"| {r['new_id']} | {r['old_id']} | `{r['old_name']}` | {r['group']} "
            f"| {r['semantic']} | {sv_s} |"
        )
    lines += [
        "",
        "## Weight archive classes（留档：hard-negative 备用）",
        "",
        "| new_id | old_id | name | value (t) |",
        "|-------:|-------:|------|----------:|",
    ]
    for r in weight_map:
        lines.append(f"| {r['new_id']} | {r['old_id']} | `{r['old_name']}` | {r['speed_value']} |")

    lines.extend(["", "## Splits", ""])
    for s in split_stats:
        lines.extend(
            [
                f"### {s['split']}",
                "",
                f"- images: {s['n_images']}",
                f"- label files written: {s['n_out_label_files']}",
                f"- images with ≥1 speed box: {s['n_images_with_speed']}",
                f"- empty negatives: {s['n_empty_negatives']}",
                f"- speed boxes kept: {s['n_speed_boxes']}"
                f"（weight 归档 {s['n_weight_boxes']}，"
                f"排除 {s['n_boxes_excluded']}，其余类丢弃 {s['n_boxes_dropped']}）",
                f"- bbox width px percentiles: `{s['bbox_width_px']}` (n={s['n_bbox_sized']})",
                f"- bbox height px percentiles: `{s['bbox_height_px']}`",
                "",
                "top speed classes:",
                "",
                "| name | count |",
                "|------|------:|",
            ]
        )
        items = sorted(s["per_class_old_id"].items(), key=lambda x: -x[1])[:15]
        for oid, cnt in items:
            lines.append(f"| `{name_by_old.get(oid, oid)}` | {cnt} |")
        if s["weight_per_class_old_id"]:
            lines += ["", "top weight classes:", "", "| name | count |", "|------|------:|"]
            witems = sorted(s["weight_per_class_old_id"].items(), key=lambda x: -x[1])[:10]
            for oid, cnt in witems:
                lines.append(f"| `{name_by_old.get(oid, oid)}` | {cnt} |")
        lines.append("")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dry_run(speed_map: list[dict], weight_map: list[dict]) -> int:
    print(f"speed classes (pl*+il*): {len(speed_map)}")
    for r in speed_map:
        print(f"  new={r['new_id']:2d} old={r['old_id']:3d} {r['old_name']}"
              f" semantic={r['semantic']} value={r['speed_value']}")
    print(f"weight classes (pm*, archived): {len(weight_map)}")
    for r in weight_map:
        print(f"  new={r['new_id']:2d} old={r['old_id']:3d} {r['old_name']} value(t)={r['speed_value']}")
    for split in SPLITS:
        src = SRC_DIR / "labels" / split
        n_speed = 0
        n_weight = 0
        for lp in src.glob("*.txt"):
            for line in lp.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                oid = int(float(line.split()[0]))
                if oid in _dry_speed_ids:
                    n_speed += 1
                elif oid in _dry_weight_ids:
                    n_weight += 1
        print(f"{split}: speed boxes ≈ {n_speed}, weight boxes ≈ {n_weight}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Only print classes and estimates")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild output directories")
    parser.add_argument("--exclude", type=Path, default=None,
                        help="JSON list of boxes to drop: [{split, image, source_line, reason}]")
    args = parser.parse_args()

    if not SRC_YAML.exists():
        print(f"missing {SRC_YAML}", file=sys.stderr)
        return 1
    if not (SRC_DIR / "labels" / "train").is_dir():
        print(f"missing source labels under {SRC_DIR}", file=sys.stderr)
        return 1

    names = load_names(SRC_YAML)
    speed_map = build_class_map(names, set(SPEED_PREFIXES))
    weight_map = build_class_map(names, set(WEIGHT_PREFIXES))
    if not speed_map:
        print("no speed classes found", file=sys.stderr)
        return 1

    global _dry_speed_ids, _dry_weight_ids
    _dry_speed_ids = {r["old_id"] for r in speed_map}
    _dry_weight_ids = {r["old_id"] for r in weight_map}

    if args.dry_run:
        return dry_run(speed_map, weight_map)

    exclude_keys = load_exclusions(args.exclude)

    prepare_out_root(OUT_MC, force=args.force)
    prepare_out_root(OUT_DET, force=args.force)
    prepare_out_root(OUT_WEIGHT, force=args.force)
    META_DIR.mkdir(parents=True, exist_ok=True)

    old_to_new_speed = {r["old_id"]: r["new_id"] for r in speed_map}
    old_to_new_weight = {r["old_id"]: r["new_id"] for r in weight_map}

    (META_DIR / "class_map.json").write_text(
        json.dumps(speed_map, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (META_DIR / "weight_class_map.json").write_text(
        json.dumps(weight_map, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"filtering speed pl*+il* ({len(speed_map)} classes) "
          f"+ weight archive pm* ({len(weight_map)} classes)")
    split_stats = []
    for split in SPLITS:
        print(f"processing {split}...")
        split_stats.append(
            process_split(
                split,
                old_to_new_speed,
                set(old_to_new_speed),
                old_to_new_weight,
                set(old_to_new_weight),
                exclude_keys,
            )
        )

    write_yaml(
        CONFIGS / "TT100K_speed.yaml",
        OUT_MC,
        speed_map,
        "# TT100K speed-limit multiclass subset (pl* max speed + il* minimum speed)\n"
        "# pm* (weight limits) are NOT speed: they are archived in data/TT100K_weight_pm\n"
        "# Generated by scripts/filter_speed_limit.py — do not edit TT100K source labels\n",
    )
    write_yaml(
        CONFIGS / "TT100K_weight_pm.yaml",
        OUT_WEIGHT,
        weight_map,
        "# TT100K weight-limit archive (pm*), NOT a speed class subset\n"
        "# Reserved for hard-negative experiments; do not use as speed positives\n"
        "# Generated by scripts/filter_speed_limit.py\n",
    )
    write_yaml(
        CONFIGS / "TT100K_speed_det.yaml",
        OUT_DET,
        [{"new_id": 0, "old_name": "speed_limit_sign"}],
        "# TT100K speed-limit detector subset (single class)\n"
        "# Generated by scripts/filter_speed_limit.py\n",
    )
    write_report(
        speed_map,
        weight_map,
        split_stats,
        META_DIR / "filter_stats.json",
        REPORTS / "speed_filter_stats.md",
        exclude_used=bool(exclude_keys),
    )

    print(f"wrote {CONFIGS / 'TT100K_speed.yaml'}")
    print(f"wrote {CONFIGS / 'TT100K_weight_pm.yaml'}")
    print(f"wrote {CONFIGS / 'TT100K_speed_det.yaml'}")
    print(f"wrote {META_DIR / 'class_map.json'}")
    print(f"wrote {META_DIR / 'weight_class_map.json'}")
    print(f"wrote {REPORTS / 'speed_filter_stats.md'}")
    for s in split_stats:
        print(
            f"  {s['split']}: speed={s['n_speed_boxes']} weight={s['n_weight_boxes']} "
            f"excluded={s['n_boxes_excluded']} dropped={s['n_boxes_dropped']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
