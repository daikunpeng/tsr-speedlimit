#!/usr/bin/env python3
"""RDK X5 tiny-classifier compile (generic hb_mapper, RGB input).

Runs inside the OpenExplorer docker container. Prepares 64x64 RGB calibration
data, writes a bayes-e config with the classifier's (x/255-0.5)/0.5 = (x-127.5)/127.5
normalization baked as data_mean_and_scale, and invokes hb_mapper.

Input ONNX: (1,3,64,64) RGB, output raw logits (no softmax).
"""

import argparse
import logging
import os
import shutil
import subprocess

logging.basicConfig(level=logging.INFO,
                    format='[%(name)s] %(levelname)s %(message)s')
logger = logging.getLogger("CLS_MAPPER")

try:
    import cv2
    import numpy as np
except ImportError:
    os.system('pip install opencv-python numpy')
    import cv2
    import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx', required=True)
    ap.add_argument('--cal-images', required=True, help='dir of crop jpgs (any size)')
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--prefix', default='cls_v3_64x64_nv12')
    ap.add_argument('--cal-num', type=int, default=50)
    ap.add_argument('--jobs', type=int, default=16)
    ap.add_argument('--opt-level', default='O3')
    ap.add_argument('--mean', type=float, default=127.5)
    ap.add_argument('--scale', type=float, default=1.0 / 127.5)
    args = ap.parse_args()

    args.onnx = os.path.abspath(args.onnx)
    args.cal_images = os.path.abspath(args.cal_images)
    args.output_dir = os.path.abspath(args.output_dir)

    try:
        subprocess.run(['hb_mapper', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("hb_mapper not available")
        return 1

    import onnxruntime as ort
    sess = ort.InferenceSession(args.onnx, providers=['CPUExecutionProvider'])
    inshape = sess.get_inputs()[0].shape
    assert len(inshape) == 4, inshape
    H, W = int(inshape[2]), int(inshape[3])
    logger.info(f"input shape {inshape}")

    # gather cal images
    imgs = [os.path.join(args.cal_images, n) for n in os.listdir(args.cal_images)
            if n.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if len(imgs) > args.cal_num:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(imgs), size=args.cal_num, replace=False)
        imgs = [imgs[i] for i in idx]
    logger.info(f"using {len(imgs)} calibration crops")

    ws = os.path.abspath('.temporary_ws_cls')
    if os.path.exists(ws):
        shutil.rmtree(ws)
    cal_dir = os.path.join(ws, 'cal_data')
    os.makedirs(cal_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    for i, p in enumerate(imgs):
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (W, H))
        t = np.transpose(img, (2, 0, 1)).astype(np.float32)  # CHW, [0,255]
        t = np.expand_dims(t, 0)  # NCHW
        t.tofile(os.path.join(cal_dir, f"{i:04d}.rgbchw"))

    out_prefix = f"{args.prefix}_{W}x{H}"
    bpu_out = os.path.join(ws, 'bpu_model_output')
    cfg = f"""model_parameters:
  onnx_model: '{args.onnx}'
  march: "bayes-e"
  layer_out_dump: False
  working_dir: '{bpu_out}'
  output_model_file_prefix: '{out_prefix}'
input_parameters:
  input_name: ""
  input_type_rt: 'nv12'
  input_type_train: 'rgb'
  input_layout_train: 'NCHW'
  norm_type: 'data_mean_and_scale'
  mean_value: {args.mean}
  scale_value: {args.scale}
calibration_parameters:
  cal_data_dir: '{cal_dir}'
  cal_data_type: 'float32'
  calibration_type: 'default'
compiler_parameters:
  jobs: {args.jobs}
  compile_mode: 'latency'
  debug: true
  optimize_level: '{args.opt_level}'
"""
    cfg_path = os.path.join(ws, 'config.yaml')
    with open(cfg_path, 'w') as f:
        f.write(cfg)

    old = os.getcwd()
    try:
        os.chdir(ws)
        cmd = "hb_mapper makertbin --config config.yaml --model-type onnx"
        logger.info(cmd)
        r = os.system(cmd)
        if r != 0:
            logger.error("conversion failed")
            return 1
    finally:
        os.chdir(old)

    src = os.path.join(bpu_out, f"{out_prefix}.bin")
    if not os.path.exists(src):
        # maybe suffix differs; list
        logger.error(f"not found {src}; dir={os.listdir(bpu_out) if os.path.exists(bpu_out) else 'n/a'}")
        return 1
    dst = os.path.join(args.output_dir, f"{out_prefix}.bin")
    shutil.move(src, dst)
    log = os.path.join(ws, "hb_mapper_makertbin.log")
    if os.path.exists(log):
        shutil.move(log, os.path.join(args.output_dir, "hb_mapper_makertbin_cls.log"))
    if os.path.exists(ws):
        shutil.rmtree(ws)
    logger.info(f"done: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
