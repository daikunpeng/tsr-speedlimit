#!/usr/bin/env python3
"""Board-side TSR accuracy eval (self-domain + TT100K).

Runs detector+classifier .bin on a directory of images, matches against
YOLO-format GT labels, and reports det recall + e2e (det+cls) metrics.

Data layout (self):
  <root>/images/test/<trip>/<stem>.png
  <root>/labels_det/test/<trip>/<stem>.txt   (single class 0)
  <root>/labels_mc/test/<trip>/<stem>.txt    (speed class via TT100K new_id)

The speed value comparison maps both the mc-label (TT100K new_id) and the
classifier prediction (v3 id) to a canonical speed name (e.g. "pl20").
"""

import argparse
import json
import os
import time

import cv2
import numpy as np

import deploy_tsr as dep


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--det', required=True)
    ap.add_argument('--cls', required=True)
    ap.add_argument('--root', required=True, help='self dataset root (images/ labels_det/ labels_mc/)')
    ap.add_argument('--speed-map', required=True, help='TT100K speed class_map.json (new_id->old_name)')
    ap.add_argument('--cls-map', required=True, help='v3 class_map.json (id->name)')
    ap.add_argument('--score-thres', type=float, default=0.5)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    speed_map = json.load(open(args.speed_map))
    newid2name = {r['new_id']: r['old_name'] for r in speed_map}
    cls_map = json.load(open(args.cls_map))
    if 'classes' in cls_map:
        cls_map = cls_map['classes']
    id2name = {int(v): k for k, v in cls_map.items()}

    det = dep.Detector(args.det, score_thres=args.score_thres)
    clf = dep.Classifier(args.cls)

    imgs_dir = os.path.join(args.root, 'images', 'test')
    det_dir = os.path.join(args.root, 'labels_det', 'test')
    mc_dir = os.path.join(args.root, 'labels_mc', 'test')

    trips = sorted(os.listdir(imgs_dir))
    n_img = 0
    n_gt = 0
    det_tp_50 = 0
    det_tp_25 = 0
    det_fp = 0
    e2e_correct = 0  # det@0.5 matched AND cls speed correct
    cls_tried = 0
    detail = []
    t0 = time.time()

    for trip in trips:
        tdir = os.path.join(imgs_dir, trip)
        if not os.path.isdir(tdir):
            continue
        for fn in sorted(os.listdir(tdir)):
            if not fn.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            stem = os.path.splitext(fn)[0]
            det_lbl = os.path.join(det_dir, trip, stem + '.txt')
            mc_lbl = os.path.join(mc_dir, trip, stem + '.txt')
            gts = []
            gt_names = []
            if os.path.exists(det_lbl):
                for line in open(det_lbl):
                    p = line.split()
                    if len(p) >= 5:
                        cx, cy, w, h = map(float, p[1:5])
                        gts.append([cx, cy, w, h])
            if os.path.exists(mc_lbl):
                for line in open(mc_lbl):
                    p = line.split()
                    if len(p) >= 5:
                        gt_names.append(newid2name.get(int(p[0]), '?'))
            else:
                gt_names = ['speed_limit_sign'] * len(gts)
            n_img += 1
            if not gts:
                continue
            n_gt += len(gts)

            bgr = cv2.imread(os.path.join(tdir, fn))
            if bgr is None:
                continue
            inp = det.pre(bgr)
            out = det.model.run(inp)
            boxes, scores, cls_ids = det.post(out)

            # convert GT normalized -> xyxy pixel
            H, W = bgr.shape[:2]
            gt_xyxy = []
            for cx, cy, w, h in gts:
                x1 = (cx - w / 2) * W; y1 = (cy - h / 2) * H
                x2 = (cx + w / 2) * W; y2 = (cy + h / 2) * H
                gt_xyxy.append([x1, y1, x2, y2])

            matched = [False] * len(gt_xyxy)
            fp = 0
            for bi in range(len(boxes)):
                best_j, best_iou = -1, 0.0
                for j, g in enumerate(gt_xyxy):
                    if matched[j]:
                        continue
                    v = iou(boxes[bi], g)
                    if v > best_iou:
                        best_iou, best_j = v, j
                if best_iou >= 0.5:
                    matched[best_j] = True
                else:
                    fp += 1
            det_fp += fp
            det_tp_50 += sum(matched)

            # e2e: for each matched GT, run cls and compare speed
            # collect matched det box -> gt name
            for j, g in enumerate(gt_xyxy):
                if not matched[j]:
                    continue
                # find the det box that matched this GT
                bbox = None
                for bi in range(len(boxes)):
                    if iou(boxes[bi], g) >= 0.5:
                        bbox = boxes[bi]
                        break
                if bbox is None:
                    continue
                x1, y1, x2, y2 = map(int, bbox)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = cv2.resize(bgr[y1:y2, x1:x2], (64, 64))
                logits = clf([crop])[0]
                pred_id = int(logits.argmax())
                pred_name = id2name.get(pred_id, '?')
                gt_name = gt_names[j] if j < len(gt_names) else '?'
                cls_tried += 1
                if pred_name == gt_name:
                    e2e_correct += 1
                detail.append({'trip': trip, 'img': fn, 'gt': gt_name, 'pred': pred_name,
                               'ok': pred_name == gt_name, 'box': [x1, y1, x2, y2]})

    dt = time.time() - t0
    print(f"images={n_img} gt_boxes={n_gt} time={dt:.1f}s ({n_img/dt:.1f} img/s)")
    print(f"det recall@0.5 = {det_tp_50}/{n_gt} = {100*det_tp_50/max(1,n_gt):.1f}%")
    print(f"det FP@0.5 = {det_fp} ({det_fp/max(1,n_img):.3f}/img)")
    print(f"e2e (det@0.5 + cls) = {e2e_correct}/{cls_tried} = {100*e2e_correct/max(1,cls_tried):.1f}%")
    if args.out:
        json.dump(detail, open(args.out, 'w'), indent=1, ensure_ascii=False)
        print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
