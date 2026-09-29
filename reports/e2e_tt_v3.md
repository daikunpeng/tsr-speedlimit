# E2E 级联评估（detector → crop → classifier）

- 生成：2026-09-24T08:05:41.500552+00:00；split=test；det conf≥0.5；crop pad=20%；匹配 IoU≥0.5
- 域：det=`data/TT100K_speed_det/labels` mc=`data/TT100K_speed_mc/labels`；src=2048x2048（size640×0.3125）；imgsz=640
- 图像输入：`/home/sicv/ws/tsr_speedlimit/data/TT100K_speed_det/images/test`（GT 始终由 det/mc root 加载）
- detector：`runs/detector/smoke1/weights/best.pt`
- classifier：`runs/classifier/v3/best.pt`（28 类，含 `other`）
- 图像 3067；GT 2490；检测框 2307

## 总体

| 指标 | 值 |
|------|----:|
| Det recall@0.50 | 86.4%（2152/2490） |
| Det FP | 155（0.051/img） |
| Det FN | 338 |
| **Cls acc（det TP 且有 GT 类）** | **97.6%**（2101/2152） |
| **E2E acc = det recall × cls acc** | **84.4%** |

## 尺寸分桶（@640 等效 px）

| 桶 | GT | det recall | cls acc | e2e acc |
|----|---:|----------:|--------:|--------:|
| <8 | 711 | 64.3% | 97.8% | 62.9% |
| 8-12 | 625 | 94.1% | 98.6% | 92.8% |
| 12-16 | 415 | 96.4% | 98.0% | 94.5% |
| 16-24 | 392 | 96.4% | 97.1% | 93.6% |
| 24-32 | 213 | 94.8% | 97.0% | 92.0% |
| 32-48 | 108 | 97.2% | 94.3% | 91.7% |
| >=48 | 26 | 84.6% | 90.9% | 76.9% |

### 部署目标区间 [12, 28) px（GT 936）

- det recall: **96.2%**（900/936）
- cls acc: **97.3%**（876/900）
- **e2e: 93.6%**

## Detector FP → classifier 输出分布（`other` 拒绝情况）


| cls | count |
|-----|------:|
| `other` | 90 |
| `pl40` | 19 |
| `pl80` | 13 |
| `pl50` | 11 |
| `pl5` | 6 |
| `pl60` | 5 |
| `pl20` | 4 |
| `pl30` | 2 |
| `il60` | 2 |
| `pl70` | 1 |
| `il100` | 1 |
| `pl120` | 1 |

- **FP 被 `other` 拒绝：90/155（58%）**
- **有效 FP（漏到速度类）：65（0.021/img）**

明细：`/home/sicv/ws/tsr_speedlimit/runs/eval/e2e_tt_v3/e2e_raw.json`

