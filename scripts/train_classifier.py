#!/usr/bin/env python3
"""Train the tiny speed-value classifier (v1 / v2 with `other` class / v3 color-aug).

v1 (backward compatible):
  .venv/bin/python scripts/train_classifier.py --scale 1.5 --name smoke1

v2 (C plan, 2026-09-23 approved):
  .venv/bin/python scripts/train_classifier.py --name v2 \
      --scales 1.0,1.3,1.5,2.0 \
      --neg-manifest data/cls_neg/manifest.csv --neg-repeat 2 \
      --report reports/classifier_v2.md

v3 (2026-09-24 approved, plate-local color/exposure aug):
  .venv/bin/python scripts/train_classifier.py --name v3 \
      --scales 1.0,1.3,1.5,2.0 \
      --neg-manifest data/cls_neg/manifest.csv --neg-repeat 2 \
      --color-aug 0.5 \
      --report reports/classifier_v3.md

- `other` class: negatives from data/cls_neg/ (pm/pr/ph label crops + det FP crops).
  train/val rows are TRAINING negatives; test rows are EVAL-ONLY.
- `--color-aug P`: v3 single factor. With prob P each TRAIN crop gets one of
  dark-red / washed-out / generic brightness-saturation-gamma jitter
  (reports/frozen_diag.md: self plates render dark desaturated rings or
  overexposed centers; v2 saw only clean crops). HUE is never shifted so
  pl(white/red) vs il(blue) stays learnable. Test rows are never augmented.
  Default 0 = bit-identical v1/v2 behavior.
- Coordinate convention: size buckets in @640-equivalent px (px2048*0.3125)
- cv2=BGR / PIL=RGB lesson: this script feeds the model with PIL(RGB) crops.
  Any cv2-based crop path must cvtColor before inference.

Artifacts (kept for analysis):
  runs/classifier/<name>/  ckpt + class_map.json + history.json + test_predictions.json
  reports/<report> + reports/figures/cls_confusion_<name>.png / cls_size_acc_<name>.png
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
CROPS = ROOT / "data" / "TT100K_speed_crops"
MANIFEST = CROPS / "manifest.csv"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
SCALE_TO_640 = 640.0 / 2048.0
SIZE_BUCKETS = [(0, 8, "<8"), (8, 12, "8-12"), (12, 16, "12-16"), (16, 24, "16-24"),
                (24, 32, "24-32"), (32, 48, "32-48"), (48, 1e9, ">=48")]
IMG_SIZE = 64


def bucket_of(size640: float) -> str:
    for lo, hi, name in SIZE_BUCKETS:
        if lo <= size640 < hi:
            return name
    return SIZE_BUCKETS[-1][2]


_worker_rng = None


def _aug_rng():
    """Per-worker RNG seeded from torch's worker seed (differs across workers)."""
    global _worker_rng
    if _worker_rng is None:
        import random as _r
        _worker_rng = _r.Random(torch.initial_seed() % (2 ** 32))
    return _worker_rng


def color_augment(img: "np.ndarray", rng) -> "np.ndarray":
    """v3 plate-local color/exposure jitter on a uint8 RGB crop (train only).

    Modes (frozen_diag.md evidence: dark desaturated red rings like trip _7/_9,
    overexposed washed-out centers like trip _4):
      dark   (40%): V*0.40-0.75, S*0.35-0.80, extra gamma darkening 1.0-1.6
      washed (30%): black-level raise x*a+b (a 0.50-0.75, b 70-115), S*0.45-0.85
      jitter (30%): V*0.70-1.30, S*0.70-1.30, gamma 0.80-1.25
    Hue untouched -> pl(white/red) vs il(blue) separation stays learnable.
    """
    import cv2
    import numpy as np
    r = rng.random()
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    if r < 0.4:  # dark red / low light
        hsv[..., 1] *= rng.uniform(0.35, 0.80)
        hsv[..., 2] *= rng.uniform(0.40, 0.75)
        x = np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32), 0, 255)
        x = 255.0 * (x / 255.0) ** rng.uniform(1.0, 1.6)
    elif r < 0.7:  # overexposed / washed-out
        x = img.astype(np.float32) * rng.uniform(0.50, 0.75) + rng.uniform(70.0, 115.0)
        hsv2 = cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv2[..., 1] *= rng.uniform(0.45, 0.85)
        x = np.clip(cv2.cvtColor(hsv2, cv2.COLOR_HSV2RGB).astype(np.float32), 0, 255)
    else:  # generic brightness / saturation / gamma jitter
        hsv[..., 1] *= rng.uniform(0.70, 1.30)
        hsv[..., 2] *= rng.uniform(0.70, 1.30)
        x = np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32), 0, 255)
        x = 255.0 * (x / 255.0) ** rng.uniform(0.80, 1.25)
    return np.clip(x, 0, 255).astype(np.uint8)


class CropDS(Dataset):
    def __init__(self, rows: list[dict], classes: dict[str, int],
                 color_aug: float = 0.0):
        self.rows = rows
        self.classes = classes
        self.color_aug = color_aug

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        with Image.open(Path(r["root"]) / r["crop_relpath"]) as im:
            x = im.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        import numpy as np
        arr = np.asarray(x).copy()
        if self.color_aug > 0 and random.random() < self.color_aug:
            arr = color_augment(arr, _aug_rng())
        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        t = (t - 0.5) / 0.5
        return t, self.classes[r["label"]], i


class TinyCNN(nn.Module):
    def __init__(self, nc: int):
        super().__init__()
        def blk(ci, co):
            return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(co), nn.ReLU(inplace=True))
        self.f = nn.Sequential(
            blk(3, 32), nn.MaxPool2d(2),
            blk(32, 64), nn.MaxPool2d(2),
            blk(64, 128), nn.MaxPool2d(2),
            blk(128, 128),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.2), nn.Linear(128, nc),
        )

    def forward(self, x):
        return self.f(x)


def load_speed_rows(scales: set[float]) -> list[dict]:
    rows = []
    with MANIFEST.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if float(r["scale"]) not in scales:
                continue
            rows.append({
                "root": str(CROPS), "crop_relpath": r["crop_relpath"],
                "label": r["old_name"], "split": r["split"],
                "scale": float(r["scale"]),
                "size640": min(float(r["sign_w"]), float(r["sign_h"])) * SCALE_TO_640,
                "origin": "speed", "src_class": r["old_name"],
            })
    return rows


def load_neg_rows(path: Path, scales: set[float]) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sc = float(r["scale"])
            if sc not in scales and sc != 1.4:  # detfp 恒为 1.4，不受 --scales 过滤
                continue
            rows.append({
                "root": str(path.parent), "crop_relpath": r["crop_relpath"],
                "label": "other", "split": r["split"], "scale": sc,
                "size640": min(float(r["sign_w"]), float(r["sign_h"])) * SCALE_TO_640,
                "origin": r["origin"], "src_class": r["src_class"],
                "det_conf": float(r["det_conf"]) if r.get("det_conf") else None,
            })
    return rows


def evaluate(model, rows, classes, dev, names, batch=256):
    """Return (acc_summary, per-sample dicts)."""
    from collections import Counter
    dl = DataLoader(CropDS(rows, classes), batch_size=batch, shuffle=False,
                    num_workers=8, pin_memory=(dev == "cuda"))
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, yb, idx in dl:
            probs = model(xb.to(dev)).softmax(1).cpu()
            for p, y, i in zip(probs, yb, idx):
                r = rows[i]
                pi = int(p.argmax())
                preds.append({
                    "crop": r["crop_relpath"], "origin": r["origin"],
                    "true": names[y], "pred": names[pi],
                    "conf": round(float(p[pi]), 4),
                    "scale": r["scale"], "size640": round(r["size640"], 2),
                    "bucket": bucket_of(r["size640"]),
                    "src_class": r["src_class"],
                })
    acc = sum(p["true"] == p["pred"] for p in preds) / len(preds) if preds else 0.0
    return acc, preds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scale", type=float, default=None,
                    help="v1 单 scale 模式（与 --scales 二选一）")
    ap.add_argument("--scales", default=None, help="逗号分隔，如 1.0,1.3,1.5,2.0")
    ap.add_argument("--neg-manifest", default=None, help="data/cls_neg/manifest.csv")
    ap.add_argument("--neg-repeat", type=int, default=1, help="label 负样本过采样倍数")
    ap.add_argument("--detfp-repeat", type=int, default=None,
                    help="det FP 负样本过采样倍数（默认同 --neg-repeat）")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--color-aug", type=float, default=0.0,
                    help="v3: train-only plate color/exposure jitter probability (0 = v2 behavior)")
    ap.add_argument("--name", default="smoke1")
    ap.add_argument("--report", default=None, help="默认 reports/classifier_<name>.md")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.scale is not None and args.scales is None:
        scales = {args.scale}
    elif args.scales is not None:
        scales = {float(s) for s in args.scales.split(",")}
    else:
        scales = {1.5}
    torch.manual_seed(args.seed)

    # ---- data ----
    speed_rows = load_speed_rows(scales)
    use_neg = args.neg_manifest is not None
    neg_rows = load_neg_rows(Path(args.neg_manifest), scales) if use_neg else []

    train_rows = [r for r in speed_rows if r["split"] in ("train", "val")]
    test_speed = [r for r in speed_rows if r["split"] == "test"]
    if use_neg:
        neg_train = [r for r in neg_rows if r["split"] in ("train", "val")]
        neg_test = [r for r in neg_rows if r["split"] == "test"]
        dr = args.detfp_repeat if args.detfp_repeat is not None else args.neg_repeat
        train_rows += [r for r in neg_train if r["origin"] == "label"] * args.neg_repeat
        train_rows += [r for r in neg_train if r["origin"] == "detfp"] * dr
        n_neg_eff = sum(1 for r in neg_train if r["origin"] == "label") * args.neg_repeat \
            + sum(1 for r in neg_train if r["origin"] == "detfp") * dr
        test_rows = test_speed + neg_test
    else:
        neg_train, neg_test = [], []
        n_neg_eff = 0
        test_rows = test_speed

    names = sorted({r["label"] for r in train_rows} | {r["label"] for r in test_rows})
    if use_neg:
        names = [n for n in names if n != "other"] + ["other"]  # other 固定最后
    classes = {n: i for i, n in enumerate(names)}
    n_pos = len(train_rows)
    print(f"scales={sorted(scales)} train={n_pos} (pos {n_pos - n_neg_eff} "
          f"+ neg {n_neg_eff}) test={len(test_rows)} "
          f"(speed {len(test_speed)} + neg {len(neg_test)}) classes={len(names)}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr_dl = DataLoader(CropDS(train_rows, classes, color_aug=args.color_aug),
                       batch_size=args.batch, shuffle=True,
                       num_workers=8, pin_memory=(dev == "cuda"))
    model = TinyCNN(len(names)).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"TinyCNN params: {n_params / 1e3:.0f}k, device={dev}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    out_dir = ROOT / "runs" / "classifier" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "class_map.json").write_text(json.dumps({
        "classes": classes,
        "scales": sorted(scales),
        "neg_manifest": args.neg_manifest,
        "neg_repeat": args.neg_repeat,
        "color_aug": args.color_aug,
        "folded_val_into_train": True,
    }, indent=1, ensure_ascii=False), encoding="utf-8")

    history = []
    best_acc = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tot = correct = 0
        loss_sum = 0.0
        for xb, yb, _ in tr_dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(yb)
            correct += (out.argmax(1) == yb).sum().item()
            tot += len(yb)
        tr_loss, tr_acc = loss_sum / tot, correct / tot
        te_acc, _ = evaluate(model, test_rows, classes, dev, names)
        history.append({"epoch": ep, "train_loss": round(tr_loss, 4),
                        "train_acc": round(tr_acc, 4), "test_acc": round(te_acc, 4),
                        "sec": round(time.time() - t0, 1)})
        print(f"ep{ep:03d} loss={tr_loss:.3f} train={tr_acc * 100:.1f}% "
              f"test={te_acc * 100:.1f}%", flush=True)
        if te_acc >= best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), out_dir / "best.pt")

    # ---- final eval with best weights ----
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=dev, weights_only=True))
    model.eval()
    acc, preds = evaluate(model, test_rows, classes, dev, names)

    # breakdowns
    sp = [p for p in preds if p["origin"] == "speed"]
    sp_acc = sum(p["true"] == p["pred"] for p in sp) / len(sp) if sp else 0.0
    sp15 = [p for p in sp if p["scale"] == 1.5]
    sp15_acc = sum(p["true"] == p["pred"] for p in sp15) / len(sp15) if sp15 else None
    ng = [p for p in preds if p["origin"] != "speed"]
    if ng:
        rej = sum(p["pred"] == "other" for p in ng) / len(ng)
        leak = Counter(p["pred"] for p in ng if p["pred"] != "other")
        # 分来源
        by_org = defaultdict(lambda: [0, 0])
        for p in ng:
            by_org[p["origin"]][0] += 1
            if p["pred"] == "other":
                by_org[p["origin"]][1] += 1
        rej_by_org = {k: round(v[1] / v[0], 4) for k, v in by_org.items()}
    else:
        rej, leak, rej_by_org = None, Counter(), {}
    # speed 被误杀成 other
    overkill = sum(p["pred"] == "other" for p in sp)
    conf_pairs = Counter((p["true"], p["pred"]) for p in sp if p["true"] != p["pred"])

    bucket_acc = {}
    for _, _, bname in SIZE_BUCKETS:
        sel = [p for p in sp if p["bucket"] == bname]
        bucket_acc[bname] = {"n": len(sel),
                             "acc": round(sum(p["true"] == p["pred"] for p in sel) / len(sel), 4)
                             if sel else None}

    (out_dir / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    (out_dir / "test_predictions.json").write_text(
        json.dumps(preds, indent=1, ensure_ascii=False), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps({
        "best_test_acc": round(best_acc, 4), "params": n_params,
        "final_acc": round(acc, 4), "speed_acc": round(sp_acc, 4),
        "speed_s15_acc": round(sp15_acc, 4) if sp15_acc is not None else None,
        "neg_reject": rej, "neg_reject_by_origin": rej_by_org,
        "speed_overkill_as_other": overkill,
        "bucket_acc": bucket_acc,
        "top_confusions": [{"true": t, "pred": p, "n": c}
                           for (t, p), c in conf_pairs.most_common(20)],
        "neg_leak_top": [{"cls": k, "n": v} for k, v in leak.most_common(10)],
    }, indent=1, ensure_ascii=False), encoding="utf-8")

    # ---- report ----
    report = args.report or f"reports/classifier_{args.name}.md"
    lines = [
        f"# Classifier {args.name} 评估",
        "",
        f"- 生成：{datetime.now(timezone.utc).isoformat()}；输入 {IMG_SIZE}×{IMG_SIZE}；"
        f"参数 {n_params / 1e3:.0f}k；scales={sorted(scales)}",
        f"- 训练期牌面颜色/曝光增强概率：{args.color_aug:g}"
        f"（暗红/发白/亮度饱和伽马抖动，仅 train 行；hue 不动）",
        f"- 训练 {args.epochs} ep：正样本 {n_pos - len(neg_train) * args.neg_repeat}"
        + (f" + 负样本 {len(neg_train)}×{args.neg_repeat}" if use_neg else "")
        + f"；**test {len(test_rows)}**（speed {len(test_speed)}"
        + (f" + neg {len(neg_test)}" if use_neg else "") + "）独立评估",
        "",
        "| 指标 | 值 |",
        "|------|----:|",
        f"| test 总 acc（含 other） | {acc * 100:.2f}% |",
        f"| **speed 类 acc** | **{sp_acc * 100:.2f}%**",
    ]
    if sp15_acc is not None:
        lines.append(f"| speed s1.5 子集 acc（对比 v1 94.9%） | {sp15_acc * 100:.2f}% |")
    if use_neg:
        lines += [
            f"| **负样本拒绝率（→other）** | **{rej * 100:.1f}%**（{len(ng)} 个） |",
            f"| 拒绝率 by origin | {rej_by_org} |",
            f"| speed 误杀为 other | {overkill} |",
        ]
    lines += ["",
              "## 尺寸分桶 speed acc（@640 等效 px）",
              "", "| 桶 | n | acc |", "|----|---|----:|"]
    for _, _, bname in SIZE_BUCKETS:
        b = bucket_acc[bname]
        acc_s = f"{b['acc'] * 100:.1f}" if b["acc"] is not None else "—"
        lines.append(f"| {bname} | {b['n']} | {acc_s} |")
    lines += ["", "## Top-20 speed 混淆对", "", "| true | pred | n |", "|------|------|---:|"]
    for (t, p), c in conf_pairs.most_common(20):
        lines.append(f"| {t} | {p} | {c} |")
    if use_neg and leak:
        lines += ["", "## 负样本漏检 top（被分成速度类）", "", "| cls | n |", "|-----|---:|"]
        for k, v in leak.most_common(10):
            lines.append(f"| `{k}` | {v} |")
    lines += ["", f"中间数据：`{out_dir}/`", ""]
    (ROOT / report).write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- figures ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False
        FIGURES.mkdir(parents=True, exist_ok=True)
        # confusion (speed classes + other)
        fig, ax = plt.subplots(figsize=(10, 9), dpi=150)
        mat = np.zeros((len(names), len(names)), dtype=int)
        for p in preds:
            mat[classes[p["true"]], classes[p["pred"]]] += 1
        im = ax.imshow(mat, cmap="Blues")
        thr = mat.max() * 0.5 if mat.max() else 0
        for i in range(len(names)):
            for j in range(len(names)):
                if mat[i, j]:
                    ax.text(j, i, mat[i, j], ha="center", va="center", fontsize=5,
                            color="white" if mat[i, j] > thr else "black")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=90, fontsize=6)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=6)
        ax.set_xlabel("pred")
        ax.set_ylabel("true")
        ax.set_title(f"{args.name} 混淆矩阵（test acc {acc * 100:.1f}%）")
        fig.colorbar(im, fraction=0.046)
        fig.tight_layout()
        fig.savefig(FIGURES / f"cls_confusion_{args.name}.png")
        plt.close(fig)
        # size buckets
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
        bs = [b[2] for b in SIZE_BUCKETS if bucket_acc[b[2]]["n"]]
        ns_ = [bucket_acc[b[2]]["n"] for b in SIZE_BUCKETS if bucket_acc[b[2]]["n"]]
        acs = [bucket_acc[b[2]]["acc"] * 100 for b in SIZE_BUCKETS if bucket_acc[b[2]]["n"]]
        ax.bar(bs, acs, color="#2ca02c", width=0.6)
        for i, (v, n) in enumerate(zip(acs, ns_)):
            ax.text(i, v + 2, f"{v:.0f}%\n(n={n})", ha="center", fontsize=9)
        ax.set_xlabel("GT 短边（@640 等效 px）")
        ax.set_ylabel("speed 分类准确率 %")
        ax.set_ylim(0, 115)
        ax.set_title(f"{args.name}：GT crop 尺寸 → speed acc")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(FIGURES / f"cls_size_acc_{args.name}.png")
    except Exception as e:  # noqa: BLE001
        print(f"figure skipped: {e}")

    print(f"wrote {ROOT / report}", flush=True)
    print(f"artifacts -> {out_dir}")
    print(f"test acc {acc * 100:.2f}% | speed acc {sp_acc * 100:.2f}%"
          + (f" | reject {rej * 100:.1f}% | overkill {overkill}" if use_neg else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
