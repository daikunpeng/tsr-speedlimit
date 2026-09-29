#!/usr/bin/env python3
"""TSR two-stage RDK X5 board deployment (detector + tiny classifier).

Self-contained (cv2, numpy, hbm_runtime). Detector: NV12 352x640 in, 6 raw
head outputs (3x{cls,box}); classifier: RGB 64x64 in, 28-class logits out.

Usage:
  python3 deploy_tsr.py --det det.bin --cls cls.bin --image in.jpg --class-map class_map.json
  python3 deploy_tsr.py ... --dir dir/ --out out.json   # batch + json dump
  python3 deploy_tsr.py ... --image in.jpg --benchmark  # loop FPS
"""

import argparse
import json
import time

import cv2
import numpy as np
import hbm_runtime


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def bgr_to_nv12_packed(bgr):
    h, w = bgr.shape[:2]
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420).reshape(-1)
    y = yuv[:h * w]
    u = yuv[h * w:h * w + h * w // 4]
    v = yuv[h * w + h * w // 4:]
    uv = np.stack((u, v), axis=-1).reshape(-1)
    return np.concatenate([y, uv]).astype(np.uint8)


class Detector:
    def __init__(self, model_path, score_thres=0.25, nms_thres=0.70, reg=16):
        self.model = hbm_runtime.HB_HBMRuntime(model_path)
        self.mname = self.model.model_names[0]
        self.input_names = self.model.input_names[self.mname]
        self.output_names = self.model.output_names[self.mname]
        self.input_shapes = self.model.input_shapes[self.mname]
        inshape = self.input_shapes[self.input_names[0]]
        if inshape[1] == 3:
            self.input_h, self.input_w = inshape[2], inshape[3]
        else:
            self.input_h, self.input_w = inshape[1], inshape[2]
        self.score_thres = score_thres
        self.nms_thres = nms_thres
        self.reg = reg
        self.strides = [8, 16, 32]
        self.weights = np.arange(reg, dtype=np.float32)[None, None, :]

    def pre(self, bgr):
        h, w = bgr.shape[:2]
        if (h, w) != (self.input_h, self.input_w):
            bgr = cv2.resize(bgr, (self.input_w, self.input_h))
        nv12 = bgr_to_nv12_packed(bgr)
        return {self.mname: {self.input_names[0]: nv12}}

    def post(self, outputs):
        raw = outputs[self.mname]
        conf_raw = -np.log(1.0 / self.score_thres - 1.0)
        dets = []
        for li, stride in enumerate(self.strides):
            cls = raw[self.output_names[li * 2]]
            box = raw[self.output_names[li * 2 + 1]]
            gh, gw = self.input_h // stride, self.input_w // stride
            cls = cls.reshape(-1, cls.shape[-1])
            maxs = cls.max(axis=1) if cls.shape[1] > 1 else cls[:, 0]
            valid = np.flatnonzero(maxs >= conf_raw)
            if valid.size == 0:
                continue
            scores = sigmoid(maxs[valid])
            ids = cls[valid].argmax(axis=1)
            # DFL decode
            bf = box.reshape(-1, box.shape[-1])[valid]
            ltrb = np.sum(softmax(bf.reshape(-1, 4, self.reg), axis=2) * self.weights, axis=2)
            x = np.tile(np.arange(gw), gh) + 0.5
            y = np.repeat(np.arange(gh), gw) + 0.5
            anchor = np.stack([x, y], axis=1)[valid]
            xyxy = np.hstack([anchor - ltrb[:, :2], anchor + ltrb[:, 2:]]) * stride
            dets.append(np.hstack([xyxy, scores[:, None], ids[:, None].astype(np.float32)]))
        if not dets:
            return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=np.int32)
        d = np.concatenate(dets, axis=0).astype(np.float32)
        fb, fs, fc = [], [], []
        for c in np.unique(d[:, 5].astype(np.int32)):
            m = d[:, 5] == c
            keep = self._nms(d[m, :4], d[m, 4])
            if keep.size:
                fb.append(d[m][keep, :4]); fs.append(d[m][keep, 4]); fc.append(np.full(keep.size, c, np.int32))
        if not fb:
            return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=np.int32)
        boxes = np.concatenate(fb); scores = np.concatenate(fs); cls = np.concatenate(fc)
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, self.input_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, self.input_h)
        return boxes, scores, cls

    @staticmethod
    def _nms(xyxy, score):
        if len(xyxy) == 0:
            return np.array([], dtype=np.int64)
        order = score.argsort()[::-1]
        keep = []
        x1, y1, x2, y2 = xyxy.T
        area = (x2 - x1) * (y2 - y1)
        while order.size:
            i = order[0]; keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
            iou = inter / (area[i] + area[order[1:]] - inter + 1e-9)
            order = order[1:][iou < 0.70]
        return np.array(keep, dtype=np.int64)


class Classifier:
    def __init__(self, model_path):
        self.model = hbm_runtime.HB_HBMRuntime(model_path)
        self.mname = self.model.model_names[0]
        self.input_names = self.model.input_names[self.mname]
        self.output_names = self.model.output_names[self.mname]

    def __call__(self, crops_bgr):
        # crops_bgr: list of (64,64,3) BGR arrays (already resized by caller)
        logits = []
        for c in crops_bgr:
            nv12 = bgr_to_nv12_packed(c)
            out = self.model.run({self.mname: {self.input_names[0]: nv12}})
            logits.append(out[self.mname][self.output_names[0]].reshape(-1))
        return np.stack(logits, axis=0) if logits else np.zeros((0, 1), np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--det', required=True)
    ap.add_argument('--cls', required=True)
    ap.add_argument('--class-map', required=True, help='json {name:id}')
    ap.add_argument('--image', default=None)
    ap.add_argument('--dir', default=None)
    ap.add_argument('--out', default=None, help='json output path for --dir')
    ap.add_argument('--score-thres', type=float, default=0.25)
    ap.add_argument('--benchmark', action='store_true')
    ap.add_argument('--frames', type=int, default=100)
    args = ap.parse_args()

    cm = json.load(open(args.class_map))
    if "classes" in cm:
        cm = cm["classes"]
    id2name = {int(v): k for k, v in cm.items()}

    det = Detector(args.det, score_thres=args.score_thres)
    clf = Classifier(args.cls)

    def infer(bgr):
        inp = det.pre(bgr)
        out = det.model.run(inp)
        boxes, scores, cls_ids = det.post(out)
        results = []
        crops = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(bgr.shape[1], x2), min(bgr.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = cv2.resize(bgr[y1:y2, x1:x2], (64, 64))
            crops.append(crop)
            results.append({'box': [int(x1), int(y1), int(x2), int(y2)],
                            'det_conf': float(scores[i])})
        if crops:
            logits = clf(crops)
            preds = logits.argmax(axis=1)
            for r, p, lg in zip(results, preds, logits):
                r['cls'] = id2name.get(int(p), str(int(p)))
                r['cls_conf'] = float(np.max(lg))
        return results

    if args.image:
        bgr = cv2.imread(args.image)
        t0 = time.time()
        r = infer(bgr)
        dt = time.time() - t0
        for x in r:
            print(f"box={x['box']} det_conf={x['det_conf']:.4f} cls={x.get('cls')} cls_conf={x.get('cls_conf', 0):.4f}")
        print(f"--- {len(r)} dets in {dt*1000:.1f} ms ---")
        if args.benchmark:
            t0 = time.time()
            for _ in range(args.frames):
                infer(bgr)
            dt = time.time() - t0
            print(f"BENCHMARK {args.frames} frames in {dt:.2f}s = {args.frames/dt:.1f} FPS")

    if args.dir:
        import os
        names = sorted([n for n in os.listdir(args.dir) if n.lower().endswith(('.jpg', '.png', '.jpeg'))])
        allres = {}
        t0 = time.time()
        for n in names:
            bgr = cv2.imread(os.path.join(args.dir, n))
            if bgr is None:
                continue
            allres[n] = infer(bgr)
        dt = time.time() - t0
        print(f"processed {len(allres)} images in {dt:.2f}s = {len(allres)/dt:.1f} img/s")
        if args.out:
            json.dump(allres, open(args.out, 'w'), indent=1)
            print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
