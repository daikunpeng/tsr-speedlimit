#!/usr/bin/env python3
"""Speed-class distribution report over TT100K speed crops.

Analyzes the two speed sign families and their speed values:
  pl* = max speed limit (white bg, red ring)
  il* = minimum speed (blue bg)
Statistics are per SIGN (one tight crop, scale=1.0), not per 4x-scaled crop.

Outputs:
  reports/class_distribution.md        tables + findings (figure links inside)
  reports/figures/group_total.png      pl vs il totals (split-stacked)
  reports/figures/speed_value_dist.png per-value distribution, pl vs il
  reports/figures/class_rank.png       all classes ranked, colored by family
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "TT100K_speed_crops" / "manifest.csv"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

PL_COLOR = "#d62728"   # red family for max-speed (white bg red ring)
IL_COLOR = "#1f77b4"   # blue for minimum speed (blue bg)


def family_of(old_name: str) -> str:
    return old_name[:2]  # pl / il


def load_signs() -> list[dict]:
    if not MANIFEST.exists():
        print(f"missing {MANIFEST} — run scripts/crop_speed_signs.py first", file=sys.stderr)
        sys.exit(1)
    signs = []
    with MANIFEST.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if float(row["scale"]) == 1.0:
                signs.append(row)
    return signs


def fig_group_total(fam_split: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
    splits = ["train", "val", "test"]
    split_colors = {"train": "#8c8c8c", "val": "#4daf4a", "test": "#984ea3"}
    for i, fam in enumerate(("pl", "il")):
        bottom = 0.0
        for sp in splits:
            v = fam_split[fam][sp]
            ax.bar(i, v, bottom=bottom, color=split_colors[sp],
                   label=sp if i == 0 else None, width=0.55)
            if v > 0:
                ax.text(i, bottom + v / 2, str(v), ha="center", va="center", fontsize=9)
            bottom += v
        ax.text(i, bottom + 60, f"{int(bottom)}", ha="center", fontsize=11, weight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["pl* 最高限速", "il* 最低限速"], fontsize=12)
    ax.set_ylabel("sign 数")
    ax.set_title("限速两大类 sign 总量（按 split 堆叠）")
    ax.legend(title="split")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_value_dist(value_split: dict, out: Path) -> None:
    values = sorted({v for v in value_split["pl"]} | {v for v in value_split["il"]})
    xs = range(len(values))
    pl_counts = [value_split["pl"].get(v, 0) for v in values]
    il_counts = [value_split["il"].get(v, 0) for v in values]
    w = 0.4
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=150)
    ax.bar([x - w / 2 for x in xs], pl_counts, width=w, color=PL_COLOR, label="pl* 最高限速")
    ax.bar([x + w / 2 for x in xs], il_counts, width=w, color=IL_COLOR, label="il* 最低限速")
    for x, c in zip(xs, pl_counts):
        if c > 0:
            ax.text(x - w / 2, c + 30, str(c), ha="center", fontsize=7.5, rotation=90)
    for x, c in zip(xs, il_counts):
        if c > 0:
            ax.text(x + w / 2, c + 30, str(c), ha="center", fontsize=7.5, rotation=90)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([str(int(v)) if v == int(v) else str(v) for v in values], fontsize=10)
    ax.set_xlabel("限速值 (km/h)")
    ax.set_ylabel("sign 数")
    ax.set_title("各限速值的 sign 分布（pl vs il）")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_class_rank(class_counts: Counter, out: Path) -> None:
    items = sorted(class_counts.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    counts = [v for _, v in items]
    colors = [PL_COLOR if n.startswith("pl") else IL_COLOR for n in names]
    fig, ax = plt.subplots(figsize=(8, 10), dpi=150)
    ax.barh(names, counts, color=colors)
    for i, c in enumerate(counts):
        ax.text(c + 15, i, str(c), va="center", fontsize=8)
    ax.set_xlabel("sign 数")
    ax.set_title("各类 sign 数量排序（红=pl 最高限速，蓝=il 最低限速）")
    ax.set_xlim(0, max(counts) * 1.12)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> int:
    signs = load_signs()
    print(f"{len(signs)} signs (scale=1.0)")

    fam_split = {f: defaultdict(int) for f in ("pl", "il")}
    value_split = {f: Counter() for f in ("pl", "il")}
    class_counts: Counter = Counter()
    class_split: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    sizes: dict[str, list[float]] = defaultdict(list)
    for s in signs:
        fam = family_of(s["old_name"])
        fam_split[fam][s["split"]] += 1
        value_split[fam][float(s["speed_value"])] += 1
        class_counts[s["old_name"]] += 1
        class_split[s["old_name"]][s["split"]] += 1
        sizes[fam].append(min(float(s["sign_w"]), float(s["sign_h"])))

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig_group_total(fam_split, FIGURES / "group_total.png")
    fig_value_dist(value_split, FIGURES / "speed_value_dist.png")
    fig_class_rank(class_counts, FIGURES / "class_rank.png")

    total_pl = sum(fam_split["pl"].values())
    total_il = sum(fam_split["il"].values())
    n_classes_pl = sum(1 for c in class_counts if c.startswith("pl"))
    n_classes_il = sum(1 for c in class_counts if c.startswith("il"))
    rare = sorted(((k, v) for k, v in class_counts.items() if v < 50),
                  key=lambda kv: kv[1])

    def p50(xs: list[float]) -> float:
        ys = sorted(xs)
        return ys[len(ys) // 2]

    lines = [
        "# 限速类别分布报告（pl 最高限速 vs il 最低限速）",
        "",
        f"生成时间：{datetime.now(timezone.utc).isoformat()}；"
        f"数据：`data/TT100K_speed_crops/manifest.csv`（scale=1.0，"
        f"**{len(signs)} signs**；crops 为其 4× scale 展开，不重复计数）",
        "",
        "## 1. 两大类总量",
        "",
        "| 类别 | sign 数 | 占比 | 类数 | P50 短边 px |",
        "|------|--------:|-----:|-----:|------------:|",
        f"| pl* 最高限速 | {total_pl} | {total_pl/len(signs)*100:.1f}% | {n_classes_pl} "
        f"| {p50(sizes['pl']):.0f} |",
        f"| il* 最低限速 | {total_il} | {total_il/len(signs)*100:.1f}% | {n_classes_il} "
        f"| {p50(sizes['il']):.0f} |",
        f"| 合计 | {len(signs)} | 100% | {n_classes_pl + n_classes_il} | — |",
        "",
        "![两大类总量](figures/group_total.png)",
        "",
        "**要点**：pl:il ≈ "
        f"{total_pl/total_il:.1f}:1，最高限速占绝对主导；最低限速（高速公路场景）"
        "样本不到 14%，且尺寸中位数更大（近距离拍摄、清晰度更好）。",
        "",
        "## 2. 各限速值分布",
        "",
        "| 限速值 (km/h) | pl sign 数 | il sign 数 |",
        "|--------------|-----------:|-----------:|",
    ]
    values = sorted(set(value_split["pl"]) | set(value_split["il"]))
    for v in values:
        lines.append(f"| {v:g} | {value_split['pl'].get(v, 0)} | {value_split['il'].get(v, 0)} |")

    pl_top = value_split["pl"].most_common(3)
    pl_top_share = sum(c for _, c in pl_top) / total_pl * 100
    lines += [
        "",
        "![限速值分布](figures/speed_value_dist.png)",
        "",
        "**要点**：",
        f"- pl 头部值集中在 40/50/80 km/h（top3 占 pl 的 {pl_top_share:.0f}%），"
        "与中国城市/快速路主流限速一致；40 为绝对峰值（1346 sign）。",
        "- il 值域窄（50–110），峰值 60 km/h，符合高速最低限速场景。",
        "- il 与 pl 数值有交集（50/60/70/80/90/100/110），但外观差异大（蓝底 vs 白底红圈），"
        "分类器应能靠颜色特征区分组别。",
        "- 罕见值（pl0/pl3/pl4/pl25/pl35/pl65 等）样本极少，长尾严重。",
        "",
        "## 3. 每类明细（按 split）",
        "",
        "| 类 | 组 | train | val | test | 合计 |",
        "|----|----|------:|-----:|-----:|-----:|",
    ]
    for name, cnt in sorted(class_counts.items(), key=lambda kv: -kv[1]):
        sp = class_split[name]
        lines.append(
            f"| {name} | {name[:2]} | {sp.get('train', 0)} | {sp.get('val', 0)} "
            f"| {sp.get('test', 0)} | {cnt} |")
    lines += [
        "",
        "![各类排序](figures/class_rank.png)",
        "",
        "## 4. 长尾与不平衡（训练注意事项）",
        "",
        f"- **sign 数 < 50 的稀有类：{len(rare)} 个** —— "
        + "、".join(f"`{k}`({v})" for k, v in rare),
        "- il70（14）、il110（21）、il50（33）等最低限速类极少；"
        "建议分类阶段合并到「速度值 + 组别」层面（如 il 类仅 7 个值），"
        "或训练时 domain/class-balanced sampling。",
        "- val split 总量太小（242 sign），分类验证应以 train/test 互补评估"
        "（见 `reports/crop_stats.md` 风险提示）。",
        "- **split 不均**：个别稀有类 sign 集中在 val 而非 train"
        "（pl15: 31/41/17、pl35: 3/17/1，按 train/val/test）——"
        "若按类做 train 映射需留意其 val 样本无法参与训练。",
        "- pl0/pl3/pl4 语义存疑（0/3/4 km/h 不在 GB 标准值域），样本仅 1–3 个，"
        "建议训练时并入相邻类或作为 other 处理（不动原始标签，仅训练映射层处理）。",
        "",
    ]
    REPORTS.mkdir(parents=True, exist_ok=True)
    out_md = REPORTS / "class_distribution.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_md}")
    print(f"figures -> {FIGURES}")
    print(f"pl={total_pl} ({n_classes_pl} classes) il={total_il} ({n_classes_il} classes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
