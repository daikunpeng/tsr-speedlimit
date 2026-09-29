# E2E 级联评估（detector → crop → classifier）

- 生成：2026-09-27T16:19:34.726874+00:00；split=test；det conf≥0.5；crop pad=20%；匹配 IoU≥0.5
- 域：det=`data/TT100K_speed_det/labels` mc=`data/TT100K_speed_mc/labels`；src=2048x2048（size640×0.3125）；imgsz=640
- 图像输入：`/home/sicv/ws/tsr_speedlimit/data/TT100K_speed_det/images/test`（GT 始终由 det/mc root 加载）
- detector：`runs/detector/tt_cropcolor1_ms033/weights/best.pt`
- classifier：`runs/classifier/v3/best.pt`（28 类，含 `other`）
- 图像 3067；GT 2490；检测框 2046

## 总体

| 指标 | 值 |
|------|----:|
| Det recall@0.50 | 76.1%（1894/2490） |
| Det FP | 152（0.050/img） |
| Det FN | 596 |
| **Cls acc（det TP 且有 GT 类）** | **97.4%**（1844/1894） |
| **E2E acc = det recall × cls acc** | **74.1%** |

## 尺寸分桶（@640 等效 px）

| 桶 | GT | det recall | cls acc | e2e acc |
|----|---:|----------:|--------:|--------:|
| <8 | 711 | 43.3% | 97.7% | 42.3% |
| 8-12 | 625 | 83.4% | 98.5% | 82.1% |
| 12-16 | 415 | 90.8% | 98.1% | 89.2% |
| 16-24 | 392 | 95.7% | 97.1% | 92.9% |
| 24-32 | 213 | 90.6% | 96.4% | 87.3% |
| 32-48 | 108 | 92.6% | 92.0% | 85.2% |
| >=48 | 26 | 76.9% | 90.0% | 69.2% |

### 部署目标区间 [12, 28) px（GT 936）

- det recall: **92.7%**（868/936）
- cls acc: **97.2%**（844/868）
- **e2e: 90.2%**

## Detector FP → classifier 输出分布（`other` 拒绝情况）


| cls | count |
|-----|------:|
| `other` | 97 |
| `pl40` | 15 |
| `pl50` | 10 |
| `pl80` | 7 |
| `pl20` | 4 |
| `pl5` | 4 |
| `pl60` | 3 |
| `il90` | 3 |
| `il60` | 3 |
| `pl30` | 2 |
| `il70` | 1 |
| `pl70` | 1 |
| `il100` | 1 |
| `pl120` | 1 |

- **FP 被 `other` 拒绝：97/152（64%）**
- **有效 FP（漏到速度类）：55（0.018/img）**

明细：`/home/sicv/ws/tsr_speedlimit/runs/eval/e2e_tt_ms033/e2e_raw.json`

