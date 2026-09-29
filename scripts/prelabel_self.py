#!/usr/bin/env python3
"""Generate YOLO-format prelabels for self-collected frames (X-AnyLabeling assist).

Pipeline (identical to production e2e, 2026-09-23):
  frame (native 640x352) -> YOLO detector conf>=0.25, imgsz=(352,640) no letterbox pad
    -> crop +20% pad -> resize 64x64 -> BGR->RGB -> v2 TinyCNN (28 cls)
    -> speed name -> index in data/self/classes.txt
    -> 'other'    -> index of 'check_me' (user adjudicates pm/ph/pr, or deletes)

Every image gets a txt (empty = no detections) so the annotator knows which
frames were preprocessed. conf defaults to 0.25 (recall 97.6% in target band)
so missed hard signs surface for review rather than hiding.

Output:
  <out>/<stem>.txt                YOLO txt, indices = classes.txt line numbers
  <out>/prelabel_summary.json     counts / conf stats

Typical use:
  .venv/bin/python scripts/prelabel_self.py --images data/self/raw_images
  # once X-AnyLabeling project exists, point --out at its labels dir and rerun:
  .venv/bin/python scripts/prelabel_self.py --out <project>/labels
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ROOT / "data" / "self" / "classes.txt"
CLS_BS = 256
FLUSH_IMGS = 256


def load_classes(path: Path) -> tuple[list[str], dict[str, int]]:
    names = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if "check_me" not in names:
        raise SystemExit(f"{path}: missing required placeholder class 'check_me'")
    return names, {n: i for i, n in enumerate(names)}


def main() -> int:
    import cv2
    import torch
    from ultralytics import YOLO

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", default="data/self/raw_images")
    ap.add_argument("--out", default="data/self/prelabels")
    ap.add_argument("--classes", default=str(CLASSES))
    ap.add_argument("--det-weights", default="runs/detector/smoke1/weights/best.pt")
    ap.add_argument("--cls-weights", default="runs/classifier/v2/best.pt")
    ap.add_argument("--cls-map", default="runs/classifier/v2/class_map.json")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--imgsz", type=int, nargs=2, default=(352, 640),
                    metavar=("H", "W"), help="native GS130WI shape; avoids letterbox pad")
    args = ap.parse_args()

    img_dir = Path(args.images)
    if not img_dir.is_dir():
        raise SystemExit(f"image dir not found: {img_dir} (put frames there first)")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names, class_idx = load_classes(Path(args.classes))
    cls_map = json.loads(Path(args.cls_map).read_text(encoding="utf-8"))["classes"]
    id2cls = {v: k for k, v in cls_map.items()}
    speed_names = set(cls_map) - {"other"}
    missing = speed_names - set(class_names)
    if missing:
        raise SystemExit(f"classes.txt missing speed classes: {sorted(missing)}")
    check_idx = class_idx["check_me"]
    unknown_idx = check_idx  # any non-speed name also lands in check_me

    spec = spec_from_file_location("tc", ROOT / "scripts" / "train_classifier.py")
    tc = module_from_spec(spec)
    spec.loader.exec_module(tc)
    spec_e = spec_from_file_location("e2e", ROOT / "scripts" / "eval_e2e.py")
    e2e = module_from_spec(spec_e)
    spec_e.loader.exec_module(e2e)
    crop_padded = e2e.crop_padded

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = tc.TinyCNN(len(cls_map)).to(dev)
    model.load_state_dict(torch.load(args.cls_weights, map_location=dev, weights_only=True))
    model.eval()

    det = YOLO(args.det_weights)
    img_files = sorted(p for p in img_dir.iterdir()
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not img_files:
        raise SystemExit(f"no images in {img_dir}")
    print(f"{len(img_files)} frames; conf>={args.conf}; imgsz={tuple(args.imgsz)}; dev={dev}",
          flush=True)

    t0 = time.time()
    pending: list[dict] = []     # per-image records awaiting classification
    crop_refs: list[tuple[int, int]] = []  # (pending_idx, box_idx)
    crop_arrs: list[np.ndarray] = []
    counter: Counter[str] = Counter()
    confs: list[float] = []
    n_boxes = 0
    n_written = 0

    def flush() -> None:
        nonlocal n_written
        if not pending:
            return
        if crop_arrs:
            batch = np.stack(crop_arrs)
            probs_all = np.zeros((len(batch), len(cls_map)), np.float32)
            with torch.no_grad():
                for i in range(0, len(batch), CLS_BS):
                    chunk = batch[i:i + CLS_BS]
                    t = torch.from_numpy(chunk).permute(0, 3, 1, 2).float().to(dev) / 255.0
                    t = (t - 0.5) / 0.5
                    probs_all[i:i + len(chunk)] = model(t).softmax(1).cpu().numpy()
            for (ri, bi), probs in zip(crop_refs, probs_all):
                pending[ri]["cls"][bi] = id2cls[int(probs.argmax())]
        for rec in pending:
            w, h = rec["w"], rec["h"]
            lines = []
            for bi in range(len(rec["xyxy"])):
                name = rec["cls"][bi]
                idx = class_idx.get(name, unknown_idx)
                x1, y1, x2, y2 = rec["xyxy"][bi]
                cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                bw, bh = (x2 - x1) / w, (y2 - y1) / h
                lines.append(f"{idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                counter[name] += 1
            (out_dir / f"{rec['stem']}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            n_written += 1
        pending.clear()
        crop_refs.clear()
        crop_arrs.clear()

    results = det.predict(
        source=str(img_dir), imgsz=list(args.imgsz), conf=args.conf, iou=args.iou,
        max_det=50, stream=True, verbose=False, device=0 if dev == "cuda" else "cpu",
    )
    n_done = 0
    for r in results:
        stem = Path(r.path).stem
        n_done += 1
        if len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float64)
            conf = r.boxes.conf.cpu().numpy()
        else:
            xyxy = np.zeros((0, 4), np.float64)
            conf = np.zeros((0,), np.float32)

        rec = {"stem": stem, "xyxy": [], "cls": [""] * len(xyxy),
               "w": int(r.orig_shape[1]), "h": int(r.orig_shape[0])}
        pi = len(pending)
        img = cv2.imread(r.path) if len(xyxy) else None
        for bi in range(len(xyxy)):
            rec["xyxy"].append([float(v) for v in xyxy[bi]])
            confs.append(float(conf[bi]))
            c = crop_padded(img, xyxy[bi]) if img is not None else None
            if c is None:
                c = np.zeros((64, 64, 3), np.uint8)
            crop_refs.append((pi, bi))
            crop_arrs.append(c)
        pending.append(rec)
        n_boxes += len(xyxy)

        if len(pending) >= FLUSH_IMGS:
            flush()
            print(f"  {n_done}/{len(img_files)} ({time.time() - t0:.0f}s)", flush=True)
    flush()

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "images_dir": str(img_dir), "out": str(out_dir),
        "det_weights": args.det_weights, "cls_weights": args.cls_weights,
        "conf": args.conf, "iou": args.iou, "imgsz": list(args.imgsz),
        "n_images": n_written, "n_boxes": n_boxes,
        "n_check_me": counter.get("check_me", 0),
        "check_me_ratio": round(counter.get("check_me", 0) / max(1, n_boxes), 3),
        "per_class": dict(counter.most_common()),
        "mean_det_conf": round(float(np.mean(confs)), 4) if confs else None,
        "elapsed_s": round(time.time() - t0, 1),
        "classes_file": str(args.classes), "n_classes": len(class_names),
    }
    (out_dir / "prelabel_summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"done: {n_written} txts, {n_boxes} boxes, "
          f"check_me {summary['n_check_me']} ({summary['check_me_ratio']:.0%}), "
          f"{summary['elapsed_s']}s -> {out_dir}", flush=True)
    print(f"summary: {out_dir / 'prelabel_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
