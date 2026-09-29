# E2E 级联评估（detector → crop → classifier）

- 生成：2026-09-24T01:17:16.893963+00:00；split=test；det conf≥0.5；crop pad=20%；匹配 IoU≥0.5
- 域：det=`data/TT100K_speed_det/labels` mc=`data/TT100K_speed_mc/labels`；src=2048x2048（size640×0.3125）；imgsz=640
- 图像输入：`/home/sicv/ws/tsr_speedlimit/data/TT100K_speed_det/images/test`（GT 始终由 det/mc root 加载）
- detector：`runs/detector/tt_smear1/weights/best.pt`
- classifier：`runs/classifier/v2/best.pt`（28 类，含 `other`）
- 图像 3067；GT 2490；检测框 2199

## 总体

| 指标 | 值 |
|------|----:|
| Det recall@0.50 | 83.1%（2069/2490） |
| Det FP | 130（0.042/img） |
| Det FN | 421 |
| **Cls acc（det TP 且有 GT 类）** | **97.4%**（2016/2069） |
| **E2E acc = det recall × cls acc** | **81.0%** |

## 尺寸分桶（@640 等效 px）

| 桶 | GT | det recall | cls acc | e2e acc |
|----|---:|----------:|--------:|--------:|
| <8 | 711 | 58.1% | 95.9% | 55.7% |
| 8-12 | 625 | 89.6% | 98.0% | 87.8% |
| 12-16 | 415 | 95.4% | 97.5% | 93.0% |
| 16-24 | 392 | 94.9% | 98.7% | 93.6% |
| 24-32 | 213 | 94.8% | 97.0% | 92.0% |
| 32-48 | 108 | 97.2% | 96.2% | 93.5% |
| >=48 | 26 | 80.8% | 100.0% | 80.8% |

### 部署目标区间 [12, 28) px（GT 936）

- det recall: **94.9%**（888/936）
- cls acc: **97.7%**（868/888）
- **e2e: 92.7%**

## Detector FP → classifier 输出分布（`other` 拒绝情况）


| cls | count |
|-----|------:|
| `other` | 68 |
| `pl50` | 17 |
| `pl80` | 12 |
| `pl40` | 10 |
| `pl30` | 7 |
| `pl5` | 5 |
| `pl60` | 4 |
| `pl20` | 2 |
| `pl70` | 2 |
| `il80` | 2 |
| `il60` | 1 |

- **FP 被 `other` 拒绝：68/130（52%）**
- **有效 FP（漏到速度类）：62（0.020/img）**

明细：`/home/sicv/ws/tsr_speedlimit/runs/eval/e2e_tt_smear1/e2e_raw.json`

