#!/usr/bin/env python3
"""B0: ONNX export + FP32 parity baseline (no RDK toolchain required).

Exports (opset + simplify aligned for later hb_mapper):
  runs/export/det_deploy_352x640.onnx   input (1,3,352,640)  board deployment shape
  runs/export/det_sq640.onnx            input (1,3,640,640)  TT100K-domain parity eval
  runs/export/cls_64.onnx               input (1,3,64,64)    raw logits (no softmax)

Classifier parity: full v2 test rows (speed 4 scales + neg test), torch vs ORT —
acc, argmax agreement, max |logits delta|.

Follow-up parity evals (run after this script):
  .venv/bin/python scripts/eval_detector_size.py \
      --weights runs/export/det_sq640.onnx --name onnx_sq640 \
      --report reports/detector_eval_size_onnx.md
  .venv/bin/python scripts/eval_e2e.py \
      --det-weights runs/export/det_sq640.onnx --cls-onnx runs/export/cls_64.onnx \
      --cls-map runs/classifier/v2/class_map.json \
      --name e2e_onnx_v2 --report reports/e2e_onnx_v2.md

Board note: ultralytics ONNX keeps box-decode in graph, NMS runs host-side in
python — for X5 the toolchain will decide NMS placement; fp32 ONNX is the
calibration input for hb_mapper.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = ROOT / "runs" / "export"
REPORT = ROOT / "reports" / "export_parity.md"
CLS_BS = 256
SPEED_SCALES = {1.0, 1.3, 1.5, 2.0}


def onnx_input_shape(path: Path) -> list[int]:
    import onnx
    m = onnx.load(str(path), load_external_data=False)
    dims = []
    for d in m.graph.input[0].type.tensor_type.shape.dim:
        dims.append(d.dim_value if d.dim_value else d.dim_param)
    return [dims[1], dims[2], dims[3]] if len(dims) >= 4 else dims


def slim(src: Path, dst: Path) -> None:
    import onnx
    import onnxslim
    m = onnx.load(str(src))
    m = onnxslim.slim(m)
    onnx.save(m, str(dst))


def export_all(det_weights: str, cls_weights: str, opset: int) -> dict:
    from ultralytics import YOLO
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = {}

    det = YOLO(det_weights)
    for tag, sz in (("det_sq640", [640, 640]), ("det_deploy_352x640", [352, 640])):
        t0 = time.time()
        p = Path(det.export(format="onnx", imgsz=sz, opset=opset, simplify=True,
                            dynamic=False, half=False, verbose=False))
        dst = EXPORT_DIR / f"{tag}.onnx"
        if p.resolve() != dst.resolve():
            shutil.move(str(p), dst)  # both exports write best.onnx — move each out
        out[tag] = {"path": str(dst), "imgsz": sz, "s": round(time.time() - t0, 1)}
        print(f"exported {tag} input={onnx_input_shape(dst)} in {out[tag]['s']}s", flush=True)

    spec = spec_from_file_location("tc", ROOT / "scripts" / "train_classifier.py")
    tc = module_from_spec(spec)
    spec.loader.exec_module(tc)
    import torch
    cm = json.loads(Path(cls_weights.replace("best.pt", "class_map.json"))
                    .read_text(encoding="utf-8"))["classes"]
    model = tc.TinyCNN(len(cm))
    model.load_state_dict(torch.load(cls_weights, map_location="cpu", weights_only=True))
    model.eval()
    t0 = time.time()
    raw = EXPORT_DIR / "_cls_64_raw.onnx"
    torch.onnx.export(
        model, torch.zeros(1, 3, 64, 64), str(raw),
        opset_version=opset, input_names=["input"], output_names=["logits"],
        do_constant_folding=True, dynamo=False,
        # batch=1 is baked without dynamic_axes; parity/e2e batch at 256.
        # hb_mapper time: re-export static batch=1 if the toolchain demands it.
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    dst = EXPORT_DIR / "cls_64.onnx"
    slim(raw, dst)
    raw.unlink()
    out["cls_64"] = {"path": str(dst), "imgsz": [64, 64], "s": round(time.time() - t0, 1)}
    print(f"exported cls_64 input={onnx_input_shape(dst)} in {out['cls_64']['s']}s", flush=True)
    return out


def cls_parity(cls_weights: str, cls_map: str) -> dict:
    from PIL import Image
    import torch
    import onnxruntime as ort

    spec = spec_from_file_location("tc", ROOT / "scripts" / "train_classifier.py")
    tc = module_from_spec(spec)
    spec.loader.exec_module(tc)

    cm = json.loads(Path(cls_map).read_text(encoding="utf-8"))["classes"]
    id2name = {v: k for k, v in cm.items()}
    speed_rows = [r for r in tc.load_speed_rows(SPEED_SCALES) if r["split"] == "test"]
    neg_rows = [r for r in tc.load_neg_rows(ROOT / "data" / "cls_neg" / "manifest.csv",
                                            SPEED_SCALES) if r["split"] == "test"]
    rows = speed_rows + neg_rows
    print(f"parity rows: {len(rows)} (speed {len(speed_rows)} + neg {len(neg_rows)})", flush=True)

    n = len(rows)
    arr = np.zeros((n, 64, 64, 3), np.uint8)
    for i, r in enumerate(rows):
        with Image.open(Path(r["root"]) / r["crop_relpath"]) as im:
            arr[i] = np.asarray(im.convert("RGB").resize((64, 64), Image.BILINEAR))
    labels = np.array([cm[r["label"]] for r in rows])

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tmodel = tc.TinyCNN(len(cm))
    tmodel.load_state_dict(torch.load(cls_weights, map_location="cpu", weights_only=True))
    tmodel.eval().to(dev)

    sess = ort.InferenceSession(
        str(EXPORT_DIR / "cls_64.onnx"),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name

    torch_logits = np.zeros((n, len(cm)), np.float32)
    ort_logits = np.zeros((n, len(cm)), np.float32)
    with torch.no_grad():
        for i in range(0, n, CLS_BS):
            x = arr[i:i + CLS_BS].astype(np.float32) / 255.0
            x = (x - 0.5) / 0.5
            t = torch.from_numpy(x.transpose(0, 3, 1, 2)).to(dev)
            torch_logits[i:i + CLS_BS] = tmodel(t).cpu().numpy()
            ort_logits[i:i + CLS_BS] = sess.run(
                None, {iname: x.transpose(0, 3, 1, 2).astype(np.float32)})[0]

    t_pred, o_pred = torch_logits.argmax(1), ort_logits.argmax(1)
    res = {
        "n_rows": n,
        "torch_acc": round(float((t_pred == labels).mean()), 4),
        "ort_acc": round(float((o_pred == labels).mean()), 4),
        "argmax_agree": round(float((t_pred == o_pred).mean()), 6),
        "max_abs_logit_delta": round(float(np.abs(torch_logits - ort_logits).max()), 6),
        "providers": sess.get_providers(),
    }
    print(f"cls parity: {res}", flush=True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--det-weights", default="runs/detector/smoke1/weights/best.pt")
    ap.add_argument("--cls-weights", default="runs/classifier/v2/best.pt")
    ap.add_argument("--cls-map", default="runs/classifier/v2/class_map.json")
    ap.add_argument("--opset", type=int, default=12,
                    help="RDK toolchain commonly accepts opset<=12")
    ap.add_argument("--skip-export", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    if not args.skip_export:
        exports = export_all(args.det_weights, args.cls_weights, args.opset)
    else:
        exports = {k: {"path": str(EXPORT_DIR / f"{k}.onnx")}
                   for k in ("det_sq640", "det_deploy_352x640", "cls_64")}
        for k, v in exports.items():
            v["imgsz"] = onnx_input_shape(Path(v["path"]))

    parity = cls_parity(args.cls_weights, args.cls_map)

    lines = [
        "# B0 ONNX 导出 + FP32 Parity",
        "",
        f"- 生成：{datetime.now(timezone.utc).isoformat()}；opset={args.opset}；simplify=onnxslim",
        f"- det 源：`{args.det_weights}`；cls 源：`{args.cls_weights}`",
        "",
        "## 导出产物",
        "",
        "| 文件 | 输入 (NCHW) | 用途 |",
        "|------|-------------|------|",
    ]
    for tag, info in exports.items():
        use = {"det_deploy_352x640": "板端部署形态（GS130WI 640×352 原生）",
               "det_sq640": "TT100K 域 parity 评估",
               "cls_64": "classifier（原始 logits）"}.get(tag, "")
        shape = info.get("imgsz")  # [C,H,W]
        lines.append(f"| `{Path(info['path']).name}` | (1,{shape[0]},{shape[1]},{shape[2]}) | {use} |")
    lines += [
        "",
        "注：ONNX 含 box decode，NMS 在 python 后处理（与 torch 路径同源）；"
        "X5 工具链接入时再定 NMS 落位。",
        "",
        "## Classifier parity（full v2 test）",
        "",
        "| 指标 | 值 |",
        "|------|----:|",
        f"| 样本数 | {parity['n_rows']} |",
        f"| torch acc | {parity['torch_acc'] * 100:.2f}% |",
        f"| ORT acc | {parity['ort_acc'] * 100:.2f}% |",
        f"| argmax 一致率 | {parity['argmax_agree'] * 100:.3f}% |",
        f"| max abs logits delta | {parity['max_abs_logit_delta']:.6f} |",
        f"| ORT providers | {', '.join(parity['providers'])} |",
        "",
        "## Detector parity（det_sq640 vs torch，分报告）",
        "",
        "- 尺寸评估：`reports/detector_eval_size_onnx.md`（对比基线 `detector_eval_size.md`）",
        "",
        "## E2E parity（onnx 全链路 vs torch 基线）",
        "",
        "- `reports/e2e_onnx_v2.md` vs `reports/e2e_v2.md`",
        "",
        f"- 导出+cls parity 总耗时 {time.time() - t0:.0f}s",
        "",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {REPORT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
