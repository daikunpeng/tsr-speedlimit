#!/usr/bin/env python3
"""TSR two-stage pipeline with direct NV12 capture from MIPI camera (hobot_vio).

Goal: avoid the cv2.imread(BGR) -> BGR->NV12 color conversion in the hot path.
The camera (via ISP) delivers packed NV12; we only resize NV12 planes to the
detector input size (640x352) and feed the BPU directly. The classifier crop
step converts only the small crop region to BGR (negligible cost).

Requires a MIPI camera on the RDK X5 (auto-detected single camera).

Usage:
  python3 deploy_tsr_cam.py --det det.bin --cls cls.bin --class-map class_map.json \
      [--src-width 1280 --src-height 1088] [--fps 30] [--frames 100] [--save frame.png]
"""

import argparse
import time

import cv2
import numpy as np

from deploy_tsr import Detector, Classifier, bgr_to_nv12_packed  # noqa: F401

try:
    from hobot_vio import libsrcampy as srcampy
except ImportError:
    from hobot_vio_rdkx5 import libsrcampy as srcampy


def nv12_resize(nv12: np.ndarray, src_w: int, src_h: int, dst_w: int, dst_h: int) -> np.ndarray:
    """Resize packed NV12 (Y + interleaved UV) via per-plane resize."""
    y = nv12[:src_w * src_h].reshape(src_h, src_w)
    uv = nv12[src_w * src_h:].reshape(src_h // 2, src_w)
    u = uv[:, 0::2]
    v = uv[:, 1::2]
    y_r = cv2.resize(y, (dst_w, dst_h))
    u_r = cv2.resize(u, (dst_w // 2, dst_h // 2))
    v_r = cv2.resize(v, (dst_w // 2, dst_h // 2))
    uv_r = np.stack([u_r, v_r], axis=-1).reshape(-1)
    return np.concatenate([y_r.reshape(-1), uv_r]).astype(np.uint8)


def nv12_to_bgr(nv12: np.ndarray, w: int, h: int) -> np.ndarray:
    y = nv12[:w * h].reshape(h, w)
    uv = nv12[w * h:].reshape(h // 2, w)
    u = np.repeat(np.repeat(uv[:, 0::2], 2, axis=0), 2, axis=1)
    v = np.repeat(np.repeat(uv[:, 1::2], 2, axis=0), 2, axis=1)
    yuv = np.stack([y, u, v], axis=-1).astype(np.uint8)
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--det', required=True)
    ap.add_argument('--cls', required=True)
    ap.add_argument('--class-map', required=True)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--frames', type=int, default=100)
    ap.add_argument('--score-thres', type=float, default=0.25)
    ap.add_argument('--save', default=None, help='save first frame as png (BGR)')
    args = ap.parse_args()

    import json
    cm = json.load(open(args.class_map))
    if 'classes' in cm:
        cm = cm['classes']
    id2name = {int(v): k for k, v in cm.items()}

    det = Detector(args.det, score_thres=args.score_thres)
    clf = Classifier(args.cls)
    dw, dh = det.input_w, det.input_h  # 640, 352

    cam = srcampy.Camera()
    ret = cam.open_cam(0, -1, args.fps, dw, dh)
    if ret:
        print(f"Error: open_cam failed ret={ret}")
        return 1

    # get_img(2, w, h) returns the ISP-downscaled frame at (w, h) directly.
    first = np.frombuffer(cam.get_img(2, dw, dh), dtype=np.uint8)
    print(f"camera downscaled frame bytes={len(first)} (expect {dw * dh * 3 // 2} = {dw}x{dh} NV12)")

    def infer_nv12(packed):
        out = det.model.run({det.mname: {det.input_names[0]: packed}})
        boxes, scores, cls_ids = det.post(out)
        results = []
        crops = []
        bgr_full = None
        if len(boxes):
            bgr_full = nv12_to_bgr(packed, dw, dh)
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(dw, x2), min(dh, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = cv2.resize(bgr_full[y1:y2, x1:x2], (64, 64))
            crops.append(crop)
            results.append({'box': [int(x1), int(y1), int(x2), int(y2)],
                            'det_conf': float(scores[i])})
        if crops:
            logits = clf(crops)
            preds = logits.argmax(axis=1)
            for r, p in zip(results, preds):
                r['cls'] = id2name.get(int(p), str(int(p)))
        return results, bgr_full

    n = 0
    t0 = time.time()
    while n < args.frames:
        raw = cam.get_img(2, dw, dh)
        if not raw:
            continue
        packed = np.frombuffer(raw, dtype=np.uint8)
        results, bgr = infer_nv12(packed)
        n += 1
        if n == 1:
            print(f"first frame: {len(results)} dets -> "
                  f"{[r.get('cls') for r in results]}")
            if args.save and bgr is not None:
                cv2.imwrite(args.save, bgr)
                print(f"saved {args.save}")
    dt = time.time() - t0
    print(f"processed {n} frames in {dt:.2f}s = {n/dt:.1f} FPS")
    cam.close_cam()
    return 0


if __name__ == '__main__':
    main()
