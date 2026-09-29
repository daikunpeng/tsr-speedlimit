# E2E 级联评估（detector → crop → classifier）

- 生成：2026-09-24T05:15:07.525568+00:00；split=test；det conf≥0.5；crop pad=20%；匹配 IoU≥0.5
- 域：det=`data/TT100K_speed_det/labels` mc=`data/TT100K_speed_mc/labels`；src=2048x2048（size640×0.3125）；imgsz=640
- 图像输入：`/home/sicv/ws/tsr_speedlimit/data/TT100K_speed_det/images/test`（GT 始终由 det/mc root 加载）
- detector：`runs/detector/tt_crop1/weights/best.pt`
- classifier：`runs/classifier/v2/best.pt`（28 类，含 `other`）
- 图像 3067；GT 2490；检测框 2188

## 总体

| 指标 | 值 |
|------|----:|
| Det recall@0.50 | 82.8%（2061/2490） |
| Det FP | 127（0.041/img） |
| Det FN | 429 |
| **Cls acc（det TP 且有 GT 类）** | **97.4%**（2008/2061） |
| **E2E acc = det recall × cls acc** | **80.6%** |

## 尺寸分桶（@640 等效 px）

| 桶 | GT | det recall | cls acc | e2e acc |
|----|---:|----------:|--------:|--------:|
| <8 | 711 | 57.9% | 96.8% | 56.1% |
| 8-12 | 625 | 91.8% | 97.9% | 89.9% |
| 12-16 | 415 | 94.2% | 98.0% | 92.3% |
| 16-24 | 392 | 92.9% | 97.8% | 90.8% |
| 24-32 | 213 | 93.4% | 96.5% | 90.1% |
| 32-48 | 108 | 91.7% | 94.9% | 87.0% |
| >=48 | 26 | 84.6% | 100.0% | 84.6% |

### 部署目标区间 [12, 28) px（GT 936）

- det recall: **93.4%**（874/936）
- cls acc: **97.6%**（853/874）
- **e2e: 91.1%**

## Detector FP → classifier 输出分布（`other` 拒绝情况）


| cls | count |
|-----|------:|
| `other` | 78 |
| `pl50` | 11 |
| `pl80` | 8 |
| `pl30` | 7 |
| `pl5` | 6 |
| `pl40` | 6 |
| `pl60` | 4 |
| `pl70` | 3 |
| `pl100` | 2 |
| `pl20` | 1 |
| `il100` | 1 |

- **FP 被 `other` 拒绝：78/127（61%）**
- **有效 FP（漏到速度类）：49（0.016/img）**

明细：`/home/sicv/ws/tsr_speedlimit/runs/eval/e2e_tt_crop1/e2e_raw.json`

