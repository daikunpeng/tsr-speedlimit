# B1-light：ORT INT8 静态量化模拟（QDQ，非 hb_mapper 精确复现）

- 生成：2026-09-23；方法：`onnxruntime.quantize_static`，QDQ、per-channel、MinMax 校准
- 校准集：detector 200 张 train 图（640×640 /255）；classifier 512 个 train/val crop（speed+neg）
- **Detect head（66 节点 `/model.24/*`）保持 FP32**（见下方踩坑），其余 backbone/neck/head-conv INT8
- 产物：`runs/export/{det_sq640_op13,cls_64_op13}_int8.onnx`；信息 `runs/export/quant_sim_info.json`

## 重要告示：本报告是量级参考，不是权威数字

- 通用 ORT QDQ INT8 **≠** X5 BPU 量化（hb_mapper 校准算法、节点处理、**X5 支持 INT16**）
- 权威恶化数字 = B1-faithful：装 RDK X5 工具链 docker → `hb_mapper` 校准 → 跑其**量化仿真 ONNX**（待用户批准数 GB 下载）
- fp32 基线已验证 opset 12/13 同值（`reports/detector_eval_size_op13.md` vs `_onnx.md`）

## 踩坑：朴素全量 INT8 → score 通道全 0、recall 0%

首版不排除 head 节点：全图 QDQ 后 box 通道 corr 仍 0.95+，但 **score 通道（ch4）全 0**
→ 永远无检出（recall@0.10/0.25/0.50 全 0.0%，FP 也 0）。
原因：Detect head 末端 Sigmoid/Mul 的 QDQ 使目标分数下溢归零。
修法：`nodes_to_exclude=[/model.24/* 66 个节点]`（head FP32，backbone/neck 仍 INT8）。
**教训：YOLO 系 INT8 必须保 head，或逐节点排查 score 分支。**

## 恶化对比（test split，conf@0.25 口径）

### Classifier（full v2 test 11731，INT8 vs 同源 fp32）

| 指标 | fp32 | INT8 | Δ |
|------|-----:|-----:|--:|
| 总 acc | 96.29% | 96.33% | +0.04pp |
| speed acc | 96.44% | 96.47% | +0.03pp |
| 负样本拒绝 | 95.48% | 95.60% | +0.12pp |
| argmax 一致率 | — | 99.80% | — |

**结论：classifier 对 INT8 不敏感**（微升为校准噪声量级）。

### Detector（test 3067 图，INT8 vs 同源 op13 fp32）

| 指标 | fp32 | INT8（head FP32） | Δ |
|------|-----:|------------------:|--:|
| recall@0.10 | 95.1%（2368） | 94.5%（2352） | −0.6pp |
| recall@0.25 | 92.4%（2301） | 91.7%（2283） | −0.7pp |
| recall@0.50 | 86.5%（2153） | 85.5%（2130） | −1.0pp |
| 目标区间 12–28px @0.25 | 97.6% | 97.3% | −0.3pp |
| FP/img @0.25 | 0.111 | 0.112 | +0.001 |

### E2E 全链路（conf≥0.5）

| 指标 | fp32（e2e_onnx_v2） | INT8（e2e_int8_sim） | Δ |
|------|--------------------:|---------------------:|--:|
| Det recall@0.5 | 86.5% | 85.4% | −1.1pp |
| Cls acc | 97.4% | 97.5% | +0.1pp |
| **E2E** | **84.2%** | **83.3%** | **−0.9pp** |
| 目标区间 e2e | 93.6% | 92.7% | −0.9pp |
| Det FP/img | 0.051 | 0.052 | +0.001 |
| 有效 FP/img | 0.020 | 0.018 | −0.002 |

## 结论

1. **INT8（head FP32）量级估算：e2e 掉约 1pp，目标区间掉约 0.9pp，classifier 基本无损**
2. 不构成阻塞；A 方案三项验收在 INT8 模拟下仍达标（拒绝率/speed acc 均不劣化）
3. 权威数字待 B1-faithful（X5 工具链 + 量化仿真 ONNX，含 INT16 选项）
4. 附带收益：INT8 模型体积 10.3MB → 5.2MB（head FP32 后），纯 backbone/neck INT8 化

## 分报告

- det：`reports/detector_eval_size_op13.md`（fp32）vs `reports/detector_eval_size_int8.md`
- e2e：`reports/e2e_onnx_v2.md`（fp32）vs `reports/e2e_int8_sim.md`

## 复现

```bash
NV=.venv/lib/python3.11/site-packages/nvidia
export LD_LIBRARY_PATH="$NV/cublas/lib:$NV/cudnn/lib:$NV/cuda_runtime/lib:$NV/cufft/lib:$NV/curand/lib:$NV/cusolver/lib:$NV/nccl/lib:$NV/nvjitlink/lib:$NV/nvtx/lib:$NV/cuda_nvrtc/lib:$NV/cuda_cupti/lib"
.venv/bin/python scripts/quant_sim.py \
    --det-model runs/export/det_sq640_op13.onnx --cls-model runs/export/cls_64_op13.onnx
# 其后按 quant_sim.py 输出的 next 命令跑 det/e2e 评估
```
