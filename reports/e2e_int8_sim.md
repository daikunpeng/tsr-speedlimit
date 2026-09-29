# E2E 级联评估（detector → crop → classifier）

- 生成：2026-09-23T08:46:13.801732+00:00；split=test；det conf≥0.5；crop pad=20%；匹配 IoU≥0.5
- detector：`runs/export/det_sq640_op13_int8.onnx`
- classifier：`runs/export/cls_64_op13_int8.onnx`（28 类，含 `other`）
- 图像 3067；GT 2490；检测框 2285

## 总体

| 指标 | 值 |
|------|----:|
| Det recall@0.50 | 85.4%（2127/2490） |
| Det FP | 158（0.052/img） |
| Det FN | 363 |
| **Cls acc（det TP 且有 GT 类）** | **97.5%**（2074/2127） |
| **E2E acc = det recall × cls acc** | **83.3%** |

## 尺寸分桶（@640 等效 px）

| 桶 | GT | det recall | cls acc | e2e acc |
|----|---:|----------:|--------:|--------:|
| <8 | 711 | 62.7% | 97.3% | 61.0% |
| 8-12 | 625 | 93.6% | 97.9% | 91.7% |
| 12-16 | 415 | 96.1% | 97.0% | 93.3% |
| 16-24 | 392 | 95.7% | 97.6% | 93.4% |
| 24-32 | 213 | 92.5% | 98.0% | 90.6% |
| 32-48 | 108 | 95.4% | 97.1% | 92.6% |
| >=48 | 26 | 84.6% | 95.5% | 80.8% |

### 部署目标区间 [12, 28) px（GT 936）

- det recall: **95.4%**（893/936）
- cls acc: **97.2%**（868/893）
- **e2e: 92.7%**

## Detector FP → classifier 输出分布（`other` 拒绝情况）


| cls | count |
|-----|------:|
| `other` | 102 |
| `pl80` | 11 |
| `pl50` | 11 |
| `pl5` | 10 |
| `pl30` | 8 |
| `pl40` | 8 |
| `pl20` | 2 |
| `pl60` | 2 |
| `pl70` | 1 |
| `il110` | 1 |
| `il60` | 1 |
| `pl15` | 1 |

- **FP 被 `other` 拒绝：102/158（65%）**
- **有效 FP（漏到速度类）：56（0.018/img）**

明细：`/home/sicv/ws/tsr_speedlimit/runs/eval/e2e_int8_sim/e2e_raw.json`

