# E2E 级联评估（detector → crop → classifier）

- 生成：2026-09-23T08:32:04.913994+00:00；split=test；det conf≥0.5；crop pad=20%；匹配 IoU≥0.5
- detector：`runs/export/det_sq640.onnx`
- classifier：`runs/export/cls_64.onnx`（28 类，含 `other`）
- 图像 3067；GT 2490；检测框 2308

## 总体

| 指标 | 值 |
|------|----:|
| Det recall@0.50 | 86.5%（2153/2490） |
| Det FP | 155（0.051/img） |
| Det FN | 337 |
| **Cls acc（det TP 且有 GT 类）** | **97.4%**（2097/2153） |
| **E2E acc = det recall × cls acc** | **84.2%** |

## 尺寸分桶（@640 等效 px）

| 桶 | GT | det recall | cls acc | e2e acc |
|----|---:|----------:|--------:|--------:|
| <8 | 711 | 64.3% | 96.7% | 62.2% |
| 8-12 | 625 | 94.2% | 98.0% | 92.3% |
| 12-16 | 415 | 96.4% | 97.2% | 93.7% |
| 16-24 | 392 | 96.4% | 97.6% | 94.1% |
| 24-32 | 213 | 94.8% | 98.0% | 93.0% |
| 32-48 | 108 | 97.2% | 96.2% | 93.5% |
| >=48 | 26 | 84.6% | 95.5% | 80.8% |

### 部署目标区间 [12, 28) px（GT 936）

- det recall: **96.2%**（900/936）
- cls acc: **97.3%**（876/900）
- **e2e: 93.6%**

## Detector FP → classifier 输出分布（`other` 拒绝情况）


| cls | count |
|-----|------:|
| `other` | 94 |
| `pl50` | 14 |
| `pl80` | 11 |
| `pl40` | 10 |
| `pl5` | 8 |
| `pl30` | 7 |
| `pl60` | 3 |
| `pl15` | 2 |
| `pl70` | 2 |
| `pl20` | 1 |
| `il110` | 1 |
| `il60` | 1 |
| `pl10` | 1 |

- **FP 被 `other` 拒绝：94/155（61%）**
- **有效 FP（漏到速度类）：61（0.020/img）**

明细：`/home/sicv/ws/tsr_speedlimit/runs/eval/e2e_onnx_v2/e2e_raw.json`

