#!/usr/bin/env python3
"""Train the tiny speed-value classifier on GT crops (scale=1.5).

- Input : data/TT100K_speed_crops/manifest.csv, rows with scale==1.5, resized 64x64
- Labels: 27 classes with instances (pl*+il*); mapping saved to runs dir
- Split : train for fitting, test for evaluation (val too small, kept out)
- Model : tiny CNN (~0.26M params), plain torch
- Coordinate convention: size buckets in @640-equivalent px (px2048*0.3125)

All intermediate artifacts kept for analysis:
  runs/classifier/<name>/  ckpt + history.json + class_map.json + test_predictions.json
  reports/classifier_smoke.md + reports/figures/cls_confusion.png + cls_size_acc.png
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
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


def bucket_of(size640: float) -> str:
    for lo, hi, name in SIZE_BUCKETS:
        if lo <= size640 < hi:
            return name
    return SIZE_BUCKETS[-1][2]


class CropDS(Dataset):
    def __init__(self, rows: list[dict], classes: dict[str, int]):
        self.rows = rows
        self.classes = classes

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        with Image.open(CROPS / r["crop_relpath"]) as im:
            x = im.convert("RGB").resize((64, 64), Image.BILINEAR)
        # PIL -> HWC uint8 -> CHW float
        t = torch.from_numpy(_np(x)).permute(2, 0, 1).float() / 255.0
        t = (t - 0.5) / 0.5
        y = self.classes[r["old_name"]]
        return t, y, i


def _np(im):
    import numpy as np
    return np.asarray(im).copy()


class TinyCNN(nn.Module):
    def __init__(self, nc: int):
        super().__init__()
        def blk(ci, co):
            return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(co), nn.ReLU(inplace=True))
        self.f = nn.Sequential(
            blk(3, 32), nn.MaxPool2d(2),          # 32x32
            blk(32, 64), nn.MaxPool2d(2),         # 16x16
            blk(64, 128), nn.MaxPool2d(2),        # 8x8
            blk(128, 128),                        # 8x8
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.2), nn.Linear(128, nc),
        )

    def forward(self, x):
        return self.f(x)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scale", type=float, default=1.5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--name", default="smoke1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    rows = []
    with MANIFEST.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if float(r["scale"]) == args.scale:
                r["size640"] = min(float(r["sign_w"]), float(r["sign_h"])) * SCALE_TO_640
                rows.append(r)
    train_rows = [r for r in rows if r["split"] == "train"]
    test_rows = [r for r in rows if r["split"] == "test"]
    val_rows = [r for r in rows if r["split"] == "val"]
    train_rows += val_rows  # val (242) folded into training; evaluation on test only
    names = sorted({r["old_name"] for r in rows})
    classes = {n: i for i, n in enumerate(names)}
    print(f"scale={args.scale}: train={len(train_rows)} test={len(test_rows)} classes={len(names)}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr_dl = DataLoader(CropDS(train_rows, classes), batch_size=args.batch, shuffle=True,
                       num_workers=8, pin_memory=(dev == "cuda"))
    te_dl = DataLoader(CropDS(test_rows, classes), batch_size=args.batch, shuffle=False,
                       num_workers=8, pin_memory=(dev == "cuda"))

    model = TinyCNN(len(names)).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"TinyCNN params: {n_params / 1e3:.0f}k, device={dev}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    out_dir = ROOT / "runs" / "classifier" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "class_map.json").write_text(
        json.dumps({"classes": classes, "folded_val_into_train": True}, indent=1), encoding="utf-8")

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

        model.eval()
        tot = correct = 0
        with torch.no_grad():
            for xb, yb, _ in te_dl:
                out = model(xb.to(dev)).cpu()
                correct += (out.argmax(1) == yb).sum().item()
                tot += len(yb)
        te_acc = correct / tot
        history.append({"epoch": ep, "train_loss": round(tr_loss, 4),
                        "train_acc": round(tr_acc, 4), "test_acc": round(te_acc, 4),
                        "sec": round(time.time() - t0, 1)})
        print(f"ep{ep:03d} loss={tr_loss:.3f} train={tr_acc * 100:.1f}% test={te_acc * 100:.1f}%")
        if te_acc >= best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), out_dir / "best.pt")

    # final detailed eval with best weights
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=dev, weights_only=True))
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, yb, idx in te_dl:
            probs = model(xb.to(dev)).softmax(1).cpu()
            for p_i, y_i, r_i in zip(probs, yb, idx):
                r = test_rows[r_i]
                preds.append({
                    "crop": r["crop_relpath"], "image": r["source_image"],
                    "true": names[y_i], "pred": names[int(p_i.argmax())],
                    "conf": round(float(p_i.max()), 4), "size640": round(r["size640"], 2),
                    "bucket": bucket_of(r["size640"]),
                })
    acc = sum(p["true"] == p["pred"] for p in preds) / len(preds)

    # confusion pairs
    conf_pairs = Counter((p["true"], p["pred"]) for p in preds if p["true"] != p["pred"])
    conf_mat = defaultdict(lambda: defaultdict(int))
    for p in preds:
        conf_mat[p["true"]][p["pred"]] += 1

    # per-bucket accuracy
    bucket_acc = {}
    for lo, hi, bname in SIZE_BUCKETS:
        sel = [p for p in preds if p["bucket"] == bname]
        bucket_acc[bname] = {
            "n": len(sel),
            "acc": round(sum(p["true"] == p["pred"] for p in sel) / len(sel), 4) if sel else None,
        }

    (out_dir / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    (out_dir / "test_predictions.json").write_text(
        json.dumps(preds, indent=1, ensure_ascii=False), encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps({"best_test_acc": round(best_acc, 4), "params": n_params,
                    "final_acc": round(acc, 4), "bucket_acc": bucket_acc,
                    "top_confusions": [
                        {"true": t, "pred": p, "n": c}
                        for (t, p), c in conf_pairs.most_common(20)]},
                   indent=1, ensure_ascii=False), encoding="utf-8")

    # report + figures
    lines = [
        "# Tiny classifier smoke 评估",
        "",
        f"- 生成：{datetime.now(timezone.utc).isoformat()}；输入 scale={args.scale} crops，64×64；"
        f"参数 {n_params / 1e3:.0f}k",
        f"- 训练 {args.epochs} ep（train {len(train_rows)} 含并入的 val；**test {len(test_rows)} 独立评估**）",
        f"- **test 总准确率：{acc * 100:.2f}%**（best: {best_acc * 100:.2f}%）",
        "",
        "## 尺寸分桶准确率（GT crop，@640 等效 px）",
        "",
        "| 桶 | n | acc |", "|----|---|----:|",
    ]
    for lo, hi, bname in SIZE_BUCKETS:
        b = bucket_acc[bname]
        if b["acc"] is None:
            lines.append(f"| {bname} | 0 | — |")
        else:
            lines.append(f"| {bname} | {b['n']} | {b['acc'] * 100:.1f} |")
    lines += ["", "## Top-20 混淆对", "", "| true | pred | n |", "|------|------|---:|"]
    for (t, p), c in conf_pairs.most_common(20):
        lines.append(f"| {t} | {p} | {c} |")
    lines += ["", f"中间数据：`{out_dir}/`（history / test_predictions / summary）", ""]
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "classifier_smoke.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False
        FIGURES.mkdir(parents=True, exist_ok=True)
        # confusion matrix
        fig, ax = plt.subplots(figsize=(9, 8), dpi=150)
        mat = np.zeros((len(names), len(names)), dtype=int)
        for t, row in conf_mat.items():
            for p, v in row.items():
                mat[classes[t], classes[p]] = v
        im = ax.imshow(mat, cmap="Blues")
        thr = mat.max() * 0.5 if mat.max() else 0
        for i in range(len(names)):
            for j in range(len(names)):
                if mat[i, j]:
                    ax.text(j, i, mat[i, j], ha="center", va="center", fontsize=6,
                            color="white" if mat[i, j] > thr else "black")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=90, fontsize=7)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("pred")
        ax.set_ylabel("true")
        ax.set_title(f"混淆矩阵（test acc {acc * 100:.1f}%）")
        fig.colorbar(im, fraction=0.046)
        fig.tight_layout()
        fig.savefig(FIGURES / "cls_confusion.png")
        plt.close(fig)
        # size buckets
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
        bs = [b[2] for b in SIZE_BUCKETS if bucket_acc[b[2]]["n"]]
        ns = [bucket_acc[b[2]]["n"] for b in SIZE_BUCKETS if bucket_acc[b[2]]["n"]]
        ac = [bucket_acc[b[2]]["acc"] * 100 for b in SIZE_BUCKETS if bucket_acc[b[2]]["n"]]
        ax.bar(bs, ac, color="#2ca02c", width=0.6)
        for i, (v, n) in enumerate(zip(ac, ns)):
            ax.text(i, v + 2, f"{v:.0f}%\n(n={n})", ha="center", fontsize=9)
        ax.set_xlabel("GT 短边（@640 等效 px）")
        ax.set_ylabel("分类准确率 %")
        ax.set_ylim(0, 115)
        ax.set_title("GT crop 尺寸 → 分类准确率（core KPI 雏形）")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(FIGURES / "cls_size_acc.png")
    except Exception as e:  # noqa: BLE001
        print(f"figure skipped: {e}")

    print(f"wrote {REPORTS / 'classifier_smoke.md'}")
    print(f"artifacts -> {out_dir}")
    print(f"test acc: {acc * 100:.2f}% (best {best_acc * 100:.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
