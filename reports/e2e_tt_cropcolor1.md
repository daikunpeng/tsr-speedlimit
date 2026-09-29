# E2E 级联评估（detector → crop → classifier）

- 生成：2026-09-24T09:52:43.135959+00:00；split=test；det conf≥0.5；crop pad=20%；匹配 IoU≥0.5
- 域：det=`data/TT100K_speed_det/labels` mc=`data/TT100K_speed_mc/labels`；src=2048x2048（size640×0.3125）；imgsz=640
- 图像输入：`/home/sicv/ws/tsr_speedlimit/data/TT100K_speed_det/images/test`（GT 始终由 det/mc root 加载）
- detector：`runs/detector/tt_cropcolor1/weights/best.pt`
- classifier：`runs/classifier/v3/best.pt`（28 类，含 `other`）
- 图像 3067；GT 2490；检测框 2067

## 总体

| 指标 | 值 |
|------|----:|
| Det recall@0.50 | 77.2%（1922/2490） |
| Det FP | 145（0.047/img） |
| Det FN | 568 |
| **Cls acc（det TP 且有 GT 类）** | **97.8%**（1880/1922） |
| **E2E acc = det recall × cls acc** | **75.5%** |

## 尺寸分桶（@640 等效 px）

| 桶 | GT | det recall | cls acc | e2e acc |
|----|---:|----------:|--------:|--------:|
| <8 | 711 | 46.1% | 98.5% | 45.4% |
| 8-12 | 625 | 83.5% | 98.9% | 82.6% |
| 12-16 | 415 | 92.3% | 98.2% | 90.6% |
| 16-24 | 392 | 94.1% | 97.0% | 91.3% |
| 24-32 | 213 | 93.9% | 97.0% | 91.1% |
| 32-48 | 108 | 92.6% | 94.0% | 87.0% |
| >=48 | 26 | 76.9% | 95.0% | 73.1% |

### 部署目标区间 [12, 28) px（GT 936）

- det recall: **93.4%**（874/936）
- cls acc: **97.4%**（851/874）
- **e2e: 90.9%**

## Detector FP → classifier 输出分布（`other` 拒绝情况）


| cls | count |
|-----|------:|
| `other` | 86 |
| `pl40` | 20 |
| `pl50` | 12 |
| `pl80` | 7 |
| `pl20` | 5 |
| `pl30` | 3 |
| `pl5` | 3 |
| `pl60` | 3 |
| `il90` | 2 |
| `pl10` | 1 |
| `pl15` | 1 |
| `il110` | 1 |
| `pl120` | 1 |

- **FP 被 `other` 拒绝：86/145（59%）**
- **有效 FP（漏到速度类）：59（0.019/img）**

明细：`/home/sicv/ws/tsr_speedlimit/runs/eval/e2e_tt_cropcolor1/e2e_raw.json`

