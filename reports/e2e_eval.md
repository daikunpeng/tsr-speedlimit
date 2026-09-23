# E2E 级联评估（detector → crop → classifier）

- 生成：2026-09-23T07:13:03.941402+00:00；split=test；det conf≥0.5；crop pad=20%；匹配 IoU≥0.5
- detector：`runs/detector/smoke1/weights/best.pt`
- classifier：`runs/classifier/smoke1/best.pt`（27 类，无 `other`）
- 图像 3067；GT 2490；检测框 2307

## 总体

| 指标 | 值 |
|------|----:|
| Det recall@0.50 | 86.4%（2152/2490） |
| Det FP | 155（0.051/img） |
| Det FN | 338 |
| **Cls acc（det TP 且有 GT 类）** | **95.6%**（2058/2152） |
| **E2E acc = det recall × cls acc** | **82.7%** |

## 尺寸分桶（@640 等效 px）

| 桶 | GT | det recall | cls acc | e2e acc |
|----|---:|----------:|--------:|--------:|
| <8 | 711 | 64.3% | 95.8% | 61.6% |
| 8-12 | 625 | 94.1% | 96.9% | 91.2% |
| 12-16 | 415 | 96.4% | 96.5% | 93.0% |
| 16-24 | 392 | 96.4% | 95.2% | 91.8% |
| 24-32 | 213 | 94.8% | 94.1% | 89.2% |
| 32-48 | 108 | 97.2% | 91.4% | 88.9% |
| >=48 | 26 | 84.6% | 81.8% | 69.2% |

### 部署目标区间 [12, 28) px（GT 936）

- det recall: **96.2%**（900/936）
- cls acc: **95.3%**（858/900）
- **e2e: 91.7%**

## Detector FP → classifier 输出分布（`other` 类基线）

（当前 classifier 无 `other` 类，FP 会被强制分到某个速度类）

| cls | count |
|-----|------:|
| `pl30` | 38 |
| `pl50` | 34 |
| `pl40` | 22 |
| `pl5` | 14 |
| `pl60` | 13 |
| `pl80` | 12 |
| `pl20` | 8 |
| `pl15` | 4 |
| `il60` | 2 |
| `pl100` | 2 |
| `pl10` | 2 |
| `pl120` | 2 |
| `pl70` | 1 |
| `pl110` | 1 |

明细：`/home/sicv/ws/tsr_speedlimit/runs/eval/e2e_test/e2e_raw.json`

