# E2E 级联评估（detector → crop → classifier）

- 生成：2026-09-24T01:20:47.625382+00:00；split=test；det conf≥0.1；crop pad=20%；匹配 IoU≥0.5
- 域：det=`data/self/labels_det` mc=`data/self/labels_mc`；src=640x352（size640×1）；imgsz=[352, 640]
- 图像输入：`/home/sicv/ws/tsr_speedlimit/data/self/images/test`（GT 始终由 det/mc root 加载）
- detector：`runs/detector/smoke1/weights/best.pt`
- classifier：`runs/classifier/v2/best.pt`（28 类，含 `other`）
- 图像 982；GT 185；检测框 161

## 总体

| 指标 | 值 |
|------|----:|
| Det recall@0.10 | 23.2%（43/185） |
| Det FP | 118（0.120/img） |
| Det FN | 142 |
| **Cls acc（det TP 且有 GT 类）** | **67.4%**（29/43） |
| **E2E acc = det recall × cls acc** | **15.7%** |

## 尺寸分桶（@640 等效 px）

| 桶 | GT | det recall | cls acc | e2e acc |
|----|---:|----------:|--------:|--------:|
| <8 | 0 | — | — | — |
| 8-12 | 24 | 12.5% | 0.0% | 0.0% |
| 12-16 | 62 | 3.2% | 50.0% | 1.6% |
| 16-24 | 62 | 19.4% | 75.0% | 14.5% |
| 24-32 | 30 | 63.3% | 68.4% | 43.3% |
| 32-48 | 4 | 100.0% | 75.0% | 75.0% |
| >=48 | 3 | 100.0% | 100.0% | 100.0% |

### 部署目标区间 [12, 28) px（GT 145）

- det recall: **17.2%**（25/145）
- cls acc: **64.0%**（16/25）
- **e2e: 11.0%**

## Detector FP → classifier 输出分布（`other` 拒绝情况）


| cls | count |
|-----|------:|
| `pl5` | 44 |
| `other` | 41 |
| `pl50` | 17 |
| `pl40` | 14 |
| `pl80` | 2 |

- **FP 被 `other` 拒绝：41/118（35%）**
- **有效 FP（漏到速度类）：77（0.078/img）**

明细：`/home/sicv/ws/tsr_speedlimit/runs/eval/e2e_self_smoke1_conf01/e2e_raw.json`

