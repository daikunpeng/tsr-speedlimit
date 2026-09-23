#!/usr/bin/env python3
"""Color/size health check over TT100K speed crops (read-only, no verdicts).

History note (2026-09-23 postmortem, kept deliberately):
  An earlier version of this script flagged "blue-dominant pl* crops" as
  suspected pl<->il mislabels. User reviewed every montage and found ALL labels
  correct: highway signs are often printed on BLUE backing boards, and night
  shots add blue color cast. The rule "center color == sign background" is
  wrong for real-road scenes. TT100K labels are ground truth; this script now
  only reports distributions and browsable montages, and NEVER recommends
  dropping data. Any data change requires explicit human confirmation.

TT100K class semantics (verified 2026-09-23):
  pl* = max speed limit (white bg + red ring, sometimes on blue backing boards)
  il* = minimum speed (BLUE sign)
  pm* = weight limit (excluded from speed subsets; archived separately)

Outputs (all informational):
  data/TT100K_speed_meta/color_flags.json    extreme-color / tiny-sign index
  reports/label_audit.md                      stats + review conclusions
  reports/audit_montages/flags_*.jpg         browsable context sheets
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
CROPS = ROOT / "data" / "TT100K_speed_crops"
MANIFEST = CROPS / "manifest.csv"
META_DIR = ROOT / "data" / "TT100K_speed_meta"
REPORTS = ROOT / "reports"
MONTAGE_DIR = REPORTS / "audit_montages"

MIN_SIGN_PX = 16.0     # below: tiny, color stats unreliable (kept in training!)
MEAN_V_MIN = 0.06      # below: pitch dark (kept in training!)
BLUE_DOM = 0.30        # blue-dominant center: blue sign OR blue backing board
MONTAGE_CELL = 128
MONTAGE_COLS = 12


def central_region(img: Image.Image, frac: float = 0.5):
    w, h = img.size
    cw, ch = max(1, int(w * frac)), max(1, int(h * frac))
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    return img.crop((x0, y0, x0 + cw, y0 + ch))


def color_fractions(img: Image.Image) -> dict:
    """Brightness-robust hue-based pixel fractions of the central region."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mx, mn = arr.max(-1), arr.min(-1)
    diff = mx - mn + 1e-6
    hue = np.where(
        mx == r, ((g - b) / diff) % 6, np.where(mx == g, (b - r) / diff + 2, (r - g) / diff + 4)
    ) * 60
    sat = diff / (mx + 1e-6)
    val = mx
    blue = (hue >= 200) & (hue <= 260) & (sat >= 0.25) & (val >= 0.12)
    blue_dk = (hue >= 200) & (hue <= 260) & (sat >= 0.18) & (val >= 0.06) & (val < 0.12)
    neutral = (sat < 0.25) & (val >= 0.12)
    n = max(1, arr.shape[0] * arr.shape[1])
    return {
        "blue_frac": float((blue.sum() + blue_dk.sum()) / n),
        "neutral_frac": float(neutral.sum() / n),
        "mean_v": float(val.mean()),
    }


def montage(name: str, entries: list[dict], out_dir: Path) -> list[Path]:
    """Contact sheets (chunked), labeled with stem + ann + class."""
    paths = []
    if not entries:
        return paths
    out_dir.mkdir(parents=True, exist_ok=True)
    for ci, chunk in enumerate(range(0, len(entries), 240)):
        sub = entries[chunk:chunk + 240]
        rows = math.ceil(len(sub) / MONTAGE_COLS)
        sheet = Image.new("RGB", (MONTAGE_COLS * MONTAGE_CELL, rows * MONTAGE_CELL),
                          (24, 24, 24))
        draw = ImageDraw.Draw(sheet)
        for i, e in enumerate(sub):
            cx, cy = (i % MONTAGE_COLS) * MONTAGE_CELL, (i // MONTAGE_COLS) * MONTAGE_CELL
            with Image.open(CROPS / e["crop_relpath"]) as im:
                im = im.convert("RGB")
                im.thumbnail((MONTAGE_CELL - 4, MONTAGE_CELL - 18))
                sheet.paste(im, (cx + 2, cy + 2))
            label = f"{e['source_image'][:-4]}_i{e['ann_index']:02d}_{e['old_name']}"
            draw.rectangle([cx, cy + MONTAGE_CELL - 16, cx + MONTAGE_CELL, cy + MONTAGE_CELL],
                           fill=(24, 24, 24))
            draw.text((cx + 4, cy + MONTAGE_CELL - 15), label, fill=(255, 220, 80))
        suffix = f"_{ci + 1}" if len(entries) > 240 else ""
        path = out_dir / f"{name}{suffix}.jpg"
        sheet.save(path, quality=90)
        paths.append(path)
    return paths


def main() -> int:
    if not MANIFEST.exists():
        print(f"missing {MANIFEST} — run scripts/crop_speed_signs.py first", file=sys.stderr)
        return 1

    entries = []
    with MANIFEST.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if float(row["scale"]) == 1.0:
                entries.append(row)
    print(f"health-checking {len(entries)} signs (tight crops)...")

    per_class: dict[str, list[dict]] = defaultdict(list)
    blue_ctx_day: list[dict] = []
    blue_ctx_night: list[dict] = []
    tiny: list[dict] = []
    dark: list[dict] = []
    for row in entries:
        sign_short = min(float(row["sign_w"]), float(row["sign_h"]))
        with Image.open(CROPS / row["crop_relpath"]) as im:
            f = color_fractions(central_region(im))
        rec = {
            "split": row["split"],
            "source_image": row["source_image"],
            "ann_index": int(row["ann_index"]),
            "old_name": row["old_name"],
            "crop_relpath": row["crop_relpath"],
            "sign_short_px": sign_short,
            **{k: round(v, 3) for k, v in f.items()},
        }
        per_class[row["old_name"]].append(rec)
        # informational flags only — these are NOT errors, all stay in training
        if sign_short < MIN_SIGN_PX:
            rec["flag"] = "tiny"
            tiny.append(rec)
        elif f["mean_v"] < MEAN_V_MIN:
            rec["flag"] = "dark"
            dark.append(rec)
        elif f["blue_frac"] >= BLUE_DOM:
            rec["flag"] = "blue_ctx_day" if f["mean_v"] >= 0.5 else "blue_ctx_night"
            (blue_ctx_day if f["mean_v"] >= 0.5 else blue_ctx_night).append(rec)

    META_DIR.mkdir(parents=True, exist_ok=True)
    (META_DIR / "color_flags.json").write_text(
        json.dumps({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "note": "informational only; flags are NOT label errors. "
                    "User review 2026-09-23 confirmed TT100K labels correct "
                    "(blue = backing boards / night cast). No data was dropped.",
            "n_audited": len(entries),
            "blue_context_day": blue_ctx_day,
            "blue_context_night": blue_ctx_night,
            "tiny": tiny,
            "dark": dark,
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    montage("flags_blue_daylight", blue_ctx_day, MONTAGE_DIR)
    montage("flags_blue_night", blue_ctx_night, MONTAGE_DIR)
    montage("flags_tiny", tiny, MONTAGE_DIR)
    montage("flags_dark", dark, MONTAGE_DIR)

    lines = [
        "# TT100K 限速子集标签审计（最终版：标注无误）",
        "",
        f"生成时间：{datetime.now(timezone.utc).isoformat()}",
        "",
        "## 结论（用户全量复核，2026-09-23）",
        "",
        "- **TT100K 原始标注经人工全量复核确认无误，未剔除任何样本**。",
        "- 类名语义：`pl*`=最高限速、`il*`=最低限速（蓝底）、`pm*`=限重（已移出 speed 子集，归档 `data/TT100K_weight_pm/`）。",
        "- 早前颜色审计报出的 P1/P2/P3（蓝底 pl/pm）全部为**审计规则误报**：",
        "  高速路牌常为**蓝底背板上印限速/限重圆标**，且夜间存在蓝偏色；",
        "  「crop 中心底色 = 标志底色」这一假设在真实道路场景不成立。",
        "- tiny（<16px）与 dark 样本**全部保留**，贴近实际部署场景，是重要训练数据。",
        "",
        "## 复盘（记录失误，避免重犯）",
        "",
        "1. **失误一（项目早期/Cursor 时代）**：把 pm 当「最低限速」、il 当「限速信息」——语义颠倒。",
        "   修正：以图像证据 + 值域 + 用户目检确认正确语义，重建全部过滤产物。",
        "2. **失误二（本次审计 v2）**：把「蓝底占比高」当作 pl↔il 错标证据，产生 607 个误报；",
        "   曾建议人工复核后剔除。用户复核后全部证伪。",
        "   教训：**视觉统计规则只能做体检，不能做标签判定；以数据自带标签为准，",
        "   任何数据改动必须先经人眼确认**。本脚本自此仅输出分布统计与浏览拼图，不做判定。",
        "",
        "## 颜色分布（供后续分析，如背景/场景研究）",
        "",
        "| 类 | n | blue P50 | neutral P50 | meanV P50 |",
        "|----|---|----------|-------------|-----------|",
    ]
    for cname in sorted(per_class, key=lambda c: -len(per_class[c])):
        rows_c = per_class[cname]
        bl = sorted(r["blue_frac"] for r in rows_c)
        nu = sorted(r["neutral_frac"] for r in rows_c)
        mv = sorted(r["mean_v"] for r in rows_c)
        lines.append(
            f"| {cname} | {len(rows_c)} | {bl[len(bl)//2]:.2f} | {nu[len(nu)//2]:.2f} "
            f"| {mv[len(mv)//2]:.2f} |")
    lines += [
        "",
        "## 信息性标记（全部保留在训练集中）",
        "",
        "| 标记 | 数量 | 说明 |",
        "|------|------|------|",
        f"| blue_ctx_day | {len(blue_ctx_day)} | 日光蓝底占比高（蓝背板/蓝底路牌，正常场景） |",
        f"| blue_ctx_night | {len(blue_ctx_night)} | 夜景蓝偏色（正常场景） |",
        f"| tiny (<{MIN_SIGN_PX:.0f}px) | {len(tiny)} | 极小目标，远距离场景，重要训练数据 |",
        f"| dark | {len(dark)} | 全黑不可判色，保留 |",
        "",
        "浏览拼图：`reports/audit_montages/flags_*.jpg`；机器可读：`data/TT100K_speed_meta/color_flags.json`",
        "",
    ]
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "label_audit.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"audited={len(entries)} blue_day={len(blue_ctx_day)} "
          f"blue_night={len(blue_ctx_night)} tiny={len(tiny)} dark={len(dark)}")
    print(f"wrote {META_DIR / 'color_flags.json'}")
    print(f"wrote {REPORTS / 'label_audit.md'}")
    print(f"montages -> {MONTAGE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
