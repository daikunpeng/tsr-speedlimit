#!/usr/bin/env python3
"""B1-light: onnxruntime INT8 PTQ simulation (ballpark degradation, NOT hb_mapper-exact).

Produces QDQ INT8 approximations of the exported FP32 models using a small
calibration set, then measures the accuracy drop with the existing harnesses:

  runs/export/det_sq640_int8.onnx
  runs/export/cls_64_int8.onnx

Follow-up evals (run after this script):
  .venv/bin/python scripts/eval_detector_size.py --weights runs/export/det_sq640_int8.onnx \
      --name int8_sq640 --report reports/detector_eval_size_int8.md
  .venv/bin/python scripts/eval_e2e.py --det-weights runs/export/det_sq640_int8.onnx \
      --cls-onnx runs/export/cls_64_int8.onnx --cls-map runs/classifier/v2/class_map.json \
      --name e2e_int8_sim --report reports/e2e_int8_sim.md

Caveats (also stated in reports/quant_sim.md):
  - generic ORT QDQ INT8 != X5 BPU quantization (hb_mapper calibration, INT16 support);
    numbers are an ORDER-OF-MAGNITUDE indicator of INT8 sensitivity only.
  - the authoritative degradation comes from the X5 toolchain's quantized-simulation
    ONNX (B1-faithful), pending toolchain install approval.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = ROOT / "runs" / "export"
CLS_BS = 256
SPEED_SCALES = {1.0, 1.3, 1.5, 2.0}


def input_name(path: Path) -> str:
    import onnx
    return onnx.load(str(path), load_external_data=False).graph.input[0].name


class _Reader:
    """onnxruntime calibration data reader over precomputed NCHW batches."""

    def __init__(self, arrays: list[np.ndarray], name: str):
        self.arrays = arrays
        self.name = name
        self.i = 0

    def get_next(self):
        if self.i >= len(self.arrays):
            return None
        x = self.arrays[self.i]
        self.i += 1
        return {self.name: x}

    def rewind(self):
        self.i = 0


def det_calib_arrays(n: int) -> list[np.ndarray]:
    from PIL import Image
    img_dir = ROOT / "data" / "TT100K_speed_det" / "images" / "train"
    files = sorted(img_dir.iterdir())
    step = max(1, len(files) // n)
    files = files[::step][:n]
    rng = np.random.default_rng(0)
    rng.shuffle(files)
    out = []
    for f in files:
        with Image.open(f) as im:
            x = np.asarray(im.convert("RGB").resize((640, 640), Image.BILINEAR))
        out.append((x.astype(np.float32) / 255.0).transpose(2, 0, 1)[None])
    print(f"det calib: {len(out)} imgs", flush=True)
    return out


def cls_calib_arrays(n: int) -> list[np.ndarray]:
    from PIL import Image
    spec = spec_from_file_location("tc", ROOT / "scripts" / "train_classifier.py")
    tc = module_from_spec(spec)
    spec.loader.exec_module(tc)
    speed = [r for r in tc.load_speed_rows(SPEED_SCALES) if r["split"] in {"train", "val"}]
    neg = [r for r in tc.load_neg_rows(ROOT / "data" / "cls_neg" / "manifest.csv",
                                       SPEED_SCALES) if r["split"] in {"train", "val"}]
    pool = speed + neg
    rng = np.random.default_rng(0)
    idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    out = []
    for i in idx:
        r = pool[int(i)]
        with Image.open(Path(r["root"]) / r["crop_relpath"]) as im:
            x = np.asarray(im.convert("RGB").resize((64, 64), Image.BILINEAR))
        y = x.astype(np.float32) / 255.0
        y = (y - 0.5) / 0.5
        out.append(y.transpose(2, 0, 1)[None])
    print(f"cls calib: {len(out)} crops (speed+neg, train/val)", flush=True)
    return out


def quantize(src: Path, dst: Path, reader: _Reader,
             nodes_to_exclude: list[str] | None = None) -> dict:
    from onnxruntime.quantization import (
        CalibrationMethod, QuantFormat, QuantType, quantize_static,
    )
    t0 = time.time()
    quantize_static(
        str(src), str(dst), reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={"WeightSymmetric": True, "ActivationSymmetric": False},
        nodes_to_exclude=nodes_to_exclude or [],
    )
    s = round(time.time() - t0, 1)
    print(f"quantized {src.name} -> {dst.name} ({dst.stat().st_size / 1e6:.1f} MB, {s}s), "
          f"excluded {len(nodes_to_exclude or [])} head nodes", flush=True)
    return {"src": str(src), "dst": str(dst),
            "size_mb": round(dst.stat().st_size / 1e6, 1), "s": s,
            "excluded_nodes": len(nodes_to_exclude or [])}


def detect_head_nodes(path: Path) -> list[str]:
    """v5/v8 Detect head (score Sigmoid etc.) collapses to 0 under naive INT8
    (2026-09-23: full-test recall 0%, ch4 all-zero); keep head FP32."""
    import onnx
    m = onnx.load(str(path), load_external_data=False)
    names = [n.name for n in m.graph.node]
    head = [n for n in names if "model.24" in n]
    if not head:  # fallback: producer subgraph of the raw output
        out = m.graph.output[0].name
        head = [n.name for n in m.graph.node if out in n.output]
    return head


def cls_acc_diff(fp32_path: Path, int8_path: Path) -> dict:
    from PIL import Image
    import onnxruntime as ort

    spec = spec_from_file_location("tc", ROOT / "scripts" / "train_classifier.py")
    tc = module_from_spec(spec)
    spec.loader.exec_module(tc)
    cm = json.loads((ROOT / "runs" / "classifier" / "v2" / "class_map.json")
                    .read_text(encoding="utf-8"))["classes"]
    rows = [r for r in tc.load_speed_rows(SPEED_SCALES) if r["split"] == "test"]
    rows += [r for r in tc.load_neg_rows(ROOT / "data" / "cls_neg" / "manifest.csv",
                                         SPEED_SCALES) if r["split"] == "test"]

    n = len(rows)
    arr = np.zeros((n, 64, 64, 3), np.uint8)
    for i, r in enumerate(rows):
        with Image.open(Path(r["root"]) / r["crop_relpath"]) as im:
            arr[i] = np.asarray(im.convert("RGB").resize((64, 64), Image.BILINEAR))
    labels = np.array([cm[r["label"]] for r in rows])
    x = arr.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    x = x.transpose(0, 3, 1, 2).astype(np.float32)

    prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    s32 = ort.InferenceSession(str(fp32_path), providers=prov)
    s8 = ort.InferenceSession(str(int8_path), providers=prov)
    n32, n8 = s32.get_inputs()[0].name, s8.get_inputs()[0].name

    def run(sess, iname):
        out = np.zeros((n, len(cm)), np.float32)
        for i in range(0, n, CLS_BS):
            out[i:i + CLS_BS] = sess.run(None, {iname: x[i:i + CLS_BS]})[0]
        return out

    l32, l8 = run(s32, n32), run(s8, n8)
    p32, p8 = l32.argmax(1), l8.argmax(1)
    speed_mask = np.array([r["origin"] == "speed" for r in rows])
    neg_mask = ~speed_mask
    res = {
        "n_rows": n,
        "fp32_acc": round(float((p32 == labels).mean()), 4),
        "int8_acc": round(float((p8 == labels).mean()), 4),
        "argmax_agree": round(float((p32 == p8).mean()), 4),
        "fp32_speed_acc": round(float((p32[speed_mask] == labels[speed_mask]).mean()), 4),
        "int8_speed_acc": round(float((p8[speed_mask] == labels[speed_mask]).mean()), 4),
        "fp32_neg_reject": round(float((p32[neg_mask] == labels[neg_mask]).mean()), 4),
        "int8_neg_reject": round(float((p8[neg_mask] == labels[neg_mask]).mean()), 4),
    }
    print(f"cls int8 diff: {res}", flush=True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--det-model", default=str(EXPORT_DIR / "det_sq640.onnx"))
    ap.add_argument("--cls-model", default=str(EXPORT_DIR / "cls_64.onnx"))
    ap.add_argument("--n-det-calib", type=int, default=200)
    ap.add_argument("--n-cls-calib", type=int, default=512)
    ap.add_argument("--skip-quantize", action="store_true")
    args = ap.parse_args()

    det_src = Path(args.det_model)
    cls_src = Path(args.cls_model)
    det_int8 = EXPORT_DIR / (det_src.stem + "_int8.onnx")
    cls_int8 = EXPORT_DIR / (cls_src.stem + "_int8.onnx")
    info = {"generated_at": datetime.now(timezone.utc).isoformat(),
            "method": "onnxruntime quantize_static QDQ per-channel MinMax"}

    if not args.skip_quantize:
        head = detect_head_nodes(det_src)
        info["det"] = quantize(det_src, det_int8,
                               _Reader(det_calib_arrays(args.n_det_calib),
                                       input_name(det_src)),
                               nodes_to_exclude=head)
        info["cls"] = quantize(cls_src, cls_int8,
                               _Reader(cls_calib_arrays(args.n_cls_calib),
                                       input_name(cls_src)))
    else:
        info["det"] = {"dst": str(det_int8)}
        info["cls"] = {"dst": str(cls_int8)}

    info["cls_acc_diff"] = cls_acc_diff(cls_src, cls_int8)

    out = EXPORT_DIR / "quant_sim_info.json"
    out.write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    print("next:", flush=True)
    print(f"  .venv/bin/python scripts/eval_detector_size.py --weights {det_int8} "
          f"--name int8_sq640 --report reports/detector_eval_size_int8.md", flush=True)
    print(f"  .venv/bin/python scripts/eval_e2e.py --det-weights {det_int8} "
          f"--cls-onnx {cls_int8} --cls-map runs/classifier/v2/class_map.json "
          f"--name e2e_int8_sim --report reports/e2e_int8_sim.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
