# E2E 级联评估（detector → crop → classifier）

- 生成：2026-09-24T08:07:47.859146+00:00；split=test；det conf≥0.5；crop pad=20%；匹配 IoU≥0.5
- 域：det=`data/TT100K_speed_det/labels` mc=`data/TT100K_speed_mc/labels`；src=2048x2048（size640×0.3125）；imgsz=640
- 图像输入：`/home/sicv/ws/tsr_speedlimit/data/TT100K_speed_det/images/test`（GT 始终由 det/mc root 加载）
- detector：`runs/detector/tt_crop1/weights/best.pt`
- classifier：`runs/classifier/v3/best.pt`（28 类，含 `other`）
- 图像 3067；GT 2490；检测框 2188

## 总体

| 指标 | 值 |
|------|----:|
| Det recall@0.50 | 82.8%（2061/2490） |
| Det FP | 127（0.041/img） |
| Det FN | 429 |
| **Cls acc（det TP 且有 GT 类）** | **97.1%**（2001/2061） |
| **E2E acc = det recall × cls acc** | **80.4%** |

## 尺寸分桶（@640 等效 px）

| 桶 | GT | det recall | cls acc | e2e acc |
|----|---:|----------:|--------:|--------:|
| <8 | 711 | 57.9% | 97.6% | 56.5% |
| 8-12 | 625 | 91.8% | 97.9% | 89.9% |
| 12-16 | 415 | 94.2% | 97.2% | 91.6% |
| 16-24 | 392 | 92.9% | 97.0% | 90.1% |
| 24-32 | 213 | 93.4% | 97.0% | 90.6% |
| 32-48 | 108 | 91.7% | 91.9% | 84.3% |
| >=48 | 26 | 84.6% | 90.9% | 76.9% |

### 部署目标区间 [12, 28) px（GT 936）

- det recall: **93.4%**（874/936）
- cls acc: **97.0%**（848/874）
- **e2e: 90.6%**

## Detector FP → classifier 输出分布（`other` 拒绝情况）


| cls | count |
|-----|------:|
| `other` | 76 |
| `pl40` | 10 |
| `pl50` | 10 |
| `pl80` | 9 |
| `pl5` | 6 |
| `pl60` | 5 |
| `pl30` | 3 |
| `pl20` | 3 |
| `pl70` | 2 |
| `pl100` | 2 |
| `il100` | 1 |

- **FP 被 `other` 拒绝：76/127（60%）**
- **有效 FP（漏到速度类）：51（0.017/img）**

明细：`/home/sicv/ws/tsr_speedlimit/runs/eval/e2e_tt_crop1v3/e2e_raw.json`

