#!/usr/bin/env python3
"""Visualize the passing self-domain sign events for eyeballing.

Read-only on data. Consumes an existing E2E raw json + event eval json (no model
inference) and renders, per passing event, one contact-sheet JPG showing the
3-frame consecutive-correct run plus a few frames before/after, with GT/det
overlays and an enlarged GT-crop inset per frame for digit readability.

Usage:
  .venv/bin/python scripts/vis_self_events.py \
      --raw runs/eval/e2e_self_crop1v3/e2e_raw.json \
      --events runs/eval/self_crop1v3/event_eval_raw.json \
      --out runs/vis/self_events_crop1v3

Outputs:
  <out>/event{id}_{trip}_{cls}.jpg   one contact sheet per passing event
  <out>/index.html                   quick browser index
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SELF_IMG = ROOT / "data" / "self" / "images" / "test"
SELF_DET = ROOT / "data" / "self" / "labels_det" / "test"

FULL_W, FULL_H = 480, 264     # downscaled full-frame panel (src 640x352)
ZOOM_SZ = 192                 # enlarged GT-crop inset panel
GRID_COLS = 3


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


def panel(img, zoom, banner):
    """Return a labelled BGR panel (full frame downscaled) with a banner strip."""
    img = cv2.resize(img, (FULL_W, FULL_H), interpolation=cv2.INTER_AREA)
    bar = np.full((26, FULL_W, 3), 20, np.uint8)
    cv2.putText(bar, banner, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def zoom_panel(img, box):
    """Nearest-neighbor enlarged crop of a GT box (honest pixel view)."""
    x1, y1, x2, y2 = box
    h, w = img.shape[:2]
    px, py = (x2 - x1) * 0.30, (y2 - y1) * 0.30
    cx1, cy1 = max(0, int(x1 - px)), max(0, int(y1 - py))
    cx2, cy2 = min(w, int(x2 + px)), min(h, int(y2 + py))
    if cx2 <= cx1 or cy2 <= cy1:
        return np.full((ZOOM_SZ, ZOOM_SZ, 3), 30, np.uint8)
    crop = img[cy1:cy2, cx1:cx2]
    crop = cv2.resize(crop, (ZOOM_SZ, ZOOM_SZ), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(crop, (0, 0), (ZOOM_SZ - 1, ZOOM_SZ - 1), (0, 220, 0), 2)
    return crop


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="runs/eval/e2e_self_crop1v3/e2e_raw.json")
    ap.add_argument("--events", default="runs/eval/self_crop1v3/event_eval_raw.json")
    ap.add_argument("--out", default="runs/vis/self_events_crop1v3")
    ap.add_argument("--before", type=int, default=3)
    ap.add_argument("--after", type=int, default=2)
    args = ap.parse_args()

    raw = json.loads((ROOT / args.raw).read_text(encoding="utf-8"))
    evd = json.loads((ROOT / args.events).read_text(encoding="utf-8"))
    rec_by_img = {r["image"]: r for r in raw["records"]}
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    events = [e for e in evd["events"] if e["status"] == "pass"]
    if not events:
        print("no passing events")
        return 0

    for e in events:
        obs = e["observations"]
        cls = e["cls"]
        # longest consecutive-correct run, take its last 3 (closest to peak)
        runs, cur = [], []
        last_correct = None
        for o in obs:
            if o["pred_cls"] == cls:
                if cur and o["frame"] - last_correct == 1:
                    cur.append(o)
                else:
                    if cur:
                        runs.append(cur)
                    cur = [o]
                last_correct = o["frame"]
        if cur:
            runs.append(cur)
        run = max(runs, key=lambda r: (len(r), max(o["size640"] for o in r))) if runs else []
        core = run[-3:] if run else []
        if not core:
            print(f"event {e['event_id']}: no correct run, skip")
            continue
        lo, hi = core[0]["frame"] - args.before, core[-1]["frame"] + args.after
        lo = max(lo, obs[0]["frame"])
        hi = min(hi, obs[-1]["frame"])
        sel = [o for o in obs if lo <= o["frame"] <= hi]

        panels, zooms = [], []
        for o in sel:
            key = o["image"]
            img = cv2.imread(str(SELF_IMG / f"{key}.png"))
            if img is None:
                raise SystemExit(f"cannot read {key}")
            # event GT box by short-side match
            box = None
            for b in load_det_boxes(key):
                if abs(b["size640"] - o["size640"]) < 0.6:
                    box = b["xyxy"]
                    break
            rec = rec_by_img.get(key)
            # GT overlays
            for b in load_det_boxes(key):
                x1, y1, x2, y2 = map(int, b["xyxy"])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 0), 2)
            # pred overlays
            if rec:
                for p in rec["preds"]:
                    x1, y1, x2, y2 = map(int, p["xyxy"])
                    if not p["matched"]:
                        color = (255, 0, 255)
                    elif p.get("cls_correct") is False:
                        color = (0, 165, 255)
                    else:
                        color = (0, 0, 255)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, f"{p['cls']} {p['cls_conf']:.2f}",
                                (x1, min(351, y2 + 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            ok = "✓" if o["pred_cls"] == cls else "✗"
            banner = (f"{o['frame']}  {o['size640']:.0f}px  GT={cls}  "
                      f"pred={o['pred_cls'] or '—'}  {ok}")
            panels.append(panel(img, None, banner))
            if box:
                zp = zoom_panel(img, box)
            else:
                zp = np.full((ZOOM_SZ, ZOOM_SZ, 3), 30, np.uint8)
            canvas = np.full((ZOOM_SZ, FULL_W, 3), 12, np.uint8)
            off = (FULL_W - ZOOM_SZ) // 2
            canvas[:, off:off + ZOOM_SZ] = zp
            zooms.append(canvas)

        # compose grids (pad to multiple of GRID_COLS with blank)
        def grid(imgs, cell_w, cell_h):
            n = len(imgs)
            pad = (-n) % GRID_COLS
            blank = np.full((cell_h, cell_w, 3), 12, np.uint8)
            imgs = imgs + [blank] * pad
            rows = []
            for i in range(0, len(imgs), GRID_COLS):
                rows.append(np.hstack(imgs[i:i + GRID_COLS]))
            return np.vstack(rows)

        full_grid = grid(panels, FULL_W, FULL_H + 26)
        zoom_grid = grid(zooms, FULL_W, ZOOM_SZ)
        gap = np.full((8, full_grid.shape[1], 3), 12, np.uint8)
        title = np.full((40, full_grid.shape[1], 3), 10, np.uint8)
        ttext = (f"event{e['event_id']}  {e['trip']}  {cls}  "
                 f"窗口 [{e['window_frames'][0]}..{e['window_frames'][-1]}]  "
                 f"正确 {e['correct_in_window']}/{len(e['window_frames'])}  |  "
                 f"绿=GT 红=TP 橙=分类错 品红=FP；放大图为 GT 区域 ×最近邻")
        cv2.putText(title, ttext, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        sheet = np.vstack([title, full_grid, gap, zoom_grid])
        fname = out / f"event{e['event_id']}_{e['trip']}_{cls}.jpg"
        cv2.imwrite(str(fname), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"{fname}  frames {lo}..{hi} ({len(sel)}) core {[o['frame'] for o in core]}")

    # index.html
    rows = "".join(
        f'<div class=c><a href="{f.name}"><img src="{f.name}" loading=lazy></a>'
        f'<div class=b>{f.stem}</div></div>'
        for f in sorted(out.glob("*.jpg"))
    )
    (out / "index.html").write_text(
        f"""<!doctype html><meta charset=utf-8><title>{out.name}</title>
<style>body{{font-family:sans-serif;background:#111;color:#eee;margin:12px}}
.g{{display:grid;grid-template-columns:repeat(auto-fill,minmax(720px,1fr));gap:14px}}
.c img{{width:100%;border:1px solid #444}}.b{{font-size:12px;padding:3px 0}}</style>
<h2>{out.name} — 绿=GT 红=TP 橙=分类错 品红=FP</h2>
<div class=g>{rows}</div>""", encoding="utf-8")
    print(f"index -> {out / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
