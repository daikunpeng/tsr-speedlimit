# E2E 级联评估（detector → crop → classifier）

- 生成：2026-09-23T14:34:44.981906+00:00；split=test；det conf≥0.5；crop pad=20%；匹配 IoU≥0.5
- 域：det=`data/self/labels_det` mc=`data/self/labels_mc`；src=640x352（size640×1）；imgsz=[352, 640]
- 图像输入：`/home/sicv/ws/tsr_speedlimit/runs/preproc/clahe/images/test`（GT 始终由 det/mc root 加载）
- detector：`runs/detector/smoke1/weights/best.pt`
- classifier：`runs/classifier/v2/best.pt`（28 类，含 `other`）
- 图像 982；GT 185；检测框 24

## 总体

| 指标 | 值 |
|------|----:|
| Det recall@0.50 | 11.4%（21/185） |
| Det FP | 3（0.003/img） |
| Det FN | 164 |
| **Cls acc（det TP 且有 GT 类）** | **71.4%**（15/21） |
| **E2E acc = det recall × cls acc** | **8.1%** |

## 尺寸分桶（@640 等效 px）

| 桶 | GT | det recall | cls acc | e2e acc |
|----|---:|----------:|--------:|--------:|
| <8 | 0 | — | — | — |
| 8-12 | 24 | 4.2% | 0.0% | 0.0% |
| 12-16 | 62 | 0.0% | — | — |
| 16-24 | 62 | 3.2% | 50.0% | 1.6% |
| 24-32 | 30 | 43.3% | 84.6% | 36.7% |
| 32-48 | 4 | 100.0% | 75.0% | 75.0% |
| >=48 | 3 | 33.3% | 0.0% | 0.0% |

### 部署目标区间 [12, 28) px（GT 145）

- det recall: **6.9%**（10/145）
- cls acc: **70.0%**（7/10）
- **e2e: 4.8%**

## Detector FP → classifier 输出分布（`other` 拒绝情况）


| cls | count |
|-----|------:|
| `other` | 3 |

- **FP 被 `other` 拒绝：3/3（100%）**
- **有效 FP（漏到速度类）：0（0.000/img）**

明细：`/home/sicv/ws/tsr_speedlimit/runs/eval/e2e_self_clahe/e2e_raw.json`

