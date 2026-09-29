#!/usr/bin/env python3
"""Export tiny classifier (TinyCNN) to static batch=1 ONNX for RDK hb_mapper.

Input: (1,3,64,64) RGB, output: raw logits (nc classes, no softmax).
Normalization is baked into the hb_mapper config (data_mean_and_scale),
NOT into this ONNX graph — the ONNX expects the already-normalized [-1,1] input.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls-weights", default="runs/classifier/v3/best.pt")
    ap.add_argument("--cls-map", default="runs/classifier/v3/class_map.json")
    ap.add_argument("--out", default="runs/rdk/cls_v3_64x64.onnx")
    ap.add_argument("--opset", type=int, default=11)
    args = ap.parse_args()

    import train_classifier as tc

    cm = json.loads((ROOT / args.cls_map).read_text(encoding="utf-8"))["classes"]
    print(f"nc = {len(cm)} classes")

    model = tc.TinyCNN(len(cm))
    ckpt = torch.load(str(ROOT / args.cls_weights), map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch.zeros(1, 3, 64, 64),
        str(out),
        opset_version=args.opset,
        input_names=["input"],
        output_names=["logits"],
        do_constant_folding=True,
        dynamo=False,
        # static batch=1 for hb_mapper
    )
    print(f"exported {out}")

    # verify
    import onnx
    m = onnx.load(str(out))
    dims = [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]
    odims = [d.dim_value for d in m.graph.output[0].type.tensor_type.shape.dim]
    print(f"input dims {dims}, output dims {odims}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
