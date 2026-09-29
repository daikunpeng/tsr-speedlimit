# E2E 级联评估（detector → crop → classifier）

- 生成：2026-09-24T11:14:41.512105+00:00；split=test；det conf≥0.5；crop pad=20%；匹配 IoU≥0.5
- 域：det=`data/TT100K_speed_det/labels` mc=`data/TT100K_speed_mc/labels`；src=2048x2048（size640×0.3125）；imgsz=640
- 图像输入：`/home/sicv/ws/tsr_speedlimit/data/TT100K_speed_det/images/test`（GT 始终由 det/mc root 加载）
- detector：`runs/detector/tt_cropcolor1/weights/best.pt`
- classifier：`runs/classifier/v3/best.pt`（28 类，含 `other`）
- 图像 3067；GT 2490；检测框 2217

## 总体

| 指标 | 值 |
|------|----:|
| Det recall@0.50 | 80.0%（1991/2490） |
| Det FP | 226（0.074/img） |
| Det FN | 499 |
| **Cls acc（det TP 且有 GT 类）** | **97.7%**（1946/1991） |
| **E2E acc = det recall × cls acc** | **78.2%** |

## 尺寸分桶（@640 等效 px）

| 桶 | GT | det recall | cls acc | e2e acc |
|----|---:|----------:|--------:|--------:|
| <8 | 711 | 48.8% | 98.3% | 48.0% |
| 8-12 | 625 | 87.2% | 98.5% | 85.9% |
| 12-16 | 415 | 94.5% | 98.2% | 92.8% |
| 16-24 | 392 | 95.9% | 97.3% | 93.4% |
| 24-32 | 213 | 96.7% | 97.6% | 94.4% |
| 32-48 | 108 | 95.4% | 93.2% | 88.9% |
| >=48 | 26 | 84.6% | 90.9% | 76.9% |

### 部署目标区间 [12, 28) px（GT 936）

- det recall: **95.4%**（893/936）
- cls acc: **97.6%**（872/893）
- **e2e: 93.2%**

## Detector FP → classifier 输出分布（`other` 拒绝情况）


| cls | count |
|-----|------:|
| `other` | 135 |
| `pl40` | 32 |
| `pl50` | 14 |
| `pl80` | 9 |
| `pl5` | 7 |
| `pl30` | 5 |
| `pl20` | 5 |
| `pl60` | 4 |
| `pl100` | 4 |
| `il110` | 2 |
| `il80` | 2 |
| `il70` | 2 |
| `pl10` | 1 |
| `il90` | 1 |
| `il60` | 1 |
| `il100` | 1 |
| `pl120` | 1 |

- **FP 被 `other` 拒绝：135/226（60%）**
- **有效 FP（漏到速度类）：91（0.030/img）**

明细：`/home/sicv/ws/tsr_speedlimit/runs/eval/e2e_tt_cropcolor1_tta/e2e_raw.json`

