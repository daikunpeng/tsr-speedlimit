# B0 ONNX 导出 + FP32 Parity

- 生成：2026-09-23T08:28:20.272536+00:00；opset=12；simplify=onnxslim
- det 源：`runs/detector/smoke1/weights/best.pt`；cls 源：`runs/classifier/v2/best.pt`

## 导出产物

| 文件 | 输入 (NCHW) | 用途 |
|------|-------------|------|
| `det_sq640.onnx` | (1,3,640,640) | TT100K 域 parity 评估 |
| `det_deploy_352x640.onnx` | (1,3,352,640) | 板端部署形态（GS130WI 640×352 原生） |
| `cls_64.onnx` | (1,3,64,64) | classifier（原始 logits，batch 动态） |

注：ONNX 含 box decode，NMS 在 python 后处理（与 torch 路径同源）；X5 工具链接入时再定 NMS 落位。

## Classifier parity（full v2 test）

| 指标 | 值 |
|------|----:|
| 样本数 | 11731 |
| torch acc | 96.28% |
| ORT acc | 96.29% |
| argmax 一致率 | 99.992% |
| max abs logits delta | 0.029798 |
| ORT providers | CUDAExecutionProvider, CPUExecutionProvider |

## Detector parity（det_sq640.onnx vs torch，test 全量 3067 图）

| 指标 | torch 基线 | ONNX | 差 |
|------|----------:|-----:|---:|
| recall@0.10 | 95.1%（2368） | 95.1%（2368） | = |
| recall@0.25 | 92.4%（2301） | 92.4%（2301） | = |
| recall@0.50 | 86.4%（2152） | 86.5%（2153） | +1 框（conf=0.5 边界抖动） |
| 目标区间 12–28px @0.25 | 97.6% | 97.6% | = |
| FP/img @0.10/0.25/0.50 | 0.209/0.111/0.051 | 0.209/0.111/0.051 | = |

分桶明细：`reports/detector_eval_size_onnx.md` vs 基线 `reports/detector_eval_size.md`。

## E2E parity（onnx 全链路 vs torch v2 基线）

| 指标 | torch（e2e_v2） | ONNX（e2e_onnx_v2） | 差 |
|------|---------------:|--------------------:|---:|
| Det recall@0.5 | 86.4%（2152） | 86.5%（2153） | +1 框 |
| Cls acc | 97.4%（2096/2152） | 97.4%（2097/2153） | = |
| **E2E** | **84.2%** | **84.2%** | **=** |
| 目标区间 e2e | 93.7% | 93.6% | −0.1pp（同一边界框） |
| 有效 FP | 61（0.020/img） | 61（0.020/img） | = |

**结论：FP32 ONNX 与 torch 完全等价**（全链路最大偏差 = conf=0.5 边界处 1 个框；
e2e 总指标 84.2% 同值）。后续量化（B1）的任何恶化可归因于量化本身，而非导出。

## 复现命令（ORT 需 CUDA12 轮子的 LD_LIBRARY_PATH）

```bash
NV=.venv/lib/python3.11/site-packages/nvidia
export LD_LIBRARY_PATH="$NV/cublas/lib:$NV/cudnn/lib:$NV/cuda_runtime/lib:$NV/cufft/lib:$NV/curand/lib:$NV/cusolver/lib:$NV/nccl/lib:$NV/nvjitlink/lib:$NV/nvtx/lib:$NV/cuda_nvrtc/lib:$NV/cuda_cupti/lib"
.venv/bin/python scripts/export_onnx.py --skip-export          # cls parity
.venv/bin/python scripts/eval_detector_size.py --weights runs/export/det_sq640.onnx \
    --name onnx_sq640 --report reports/detector_eval_size_onnx.md
.venv/bin/python scripts/eval_e2e.py --det-weights runs/export/det_sq640.onnx \
    --cls-onnx runs/export/cls_64.onnx --cls-map runs/classifier/v2/class_map.json \
    --name e2e_onnx_v2 --report reports/e2e_onnx_v2.md
```

注：onnxruntime-gpu 需 `==1.19.2`（CUDA12 构建；1.23 要 libcublasLt.so.13 不存在于本机）。
原定「导出+cls parity 总耗时 5s」仅指 CPU 首跑；GPU parity ~40s。

