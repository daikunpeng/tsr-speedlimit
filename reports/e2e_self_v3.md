# E2E 级联评估（detector → crop → classifier）

- 生成：2026-09-24T08:02:57.846083+00:00；split=test；det conf≥0.5；crop pad=20%；匹配 IoU≥0.5
- 域：det=`data/self/labels_det` mc=`data/self/labels_mc`；src=640x352（size640×1）；imgsz=[352, 640]
- 图像输入：`/home/sicv/ws/tsr_speedlimit/data/self/images/test`（GT 始终由 det/mc root 加载）
- detector：`runs/detector/smoke1/weights/best.pt`
- classifier：`runs/classifier/v3/best.pt`（28 类，含 `other`）
- 图像 982；GT 185；检测框 29

## 总体

| 指标 | 值 |
|------|----:|
| Det recall@0.50 | 11.9%（22/185） |
| Det FP | 7（0.007/img） |
| Det FN | 163 |
| **Cls acc（det TP 且有 GT 类）** | **90.9%**（20/22） |
| **E2E acc = det recall × cls acc** | **10.8%** |

## 尺寸分桶（@640 等效 px）

| 桶 | GT | det recall | cls acc | e2e acc |
|----|---:|----------:|--------:|--------:|
| <8 | 0 | — | — | — |
| 8-12 | 24 | 8.3% | 100.0% | 8.3% |
| 12-16 | 62 | 0.0% | — | — |
| 16-24 | 62 | 3.2% | 100.0% | 3.2% |
| 24-32 | 30 | 43.3% | 84.6% | 36.7% |
| 32-48 | 4 | 100.0% | 100.0% | 100.0% |
| >=48 | 3 | 33.3% | 100.0% | 33.3% |

### 部署目标区间 [12, 28) px（GT 145）

- det recall: **6.2%**（9/145）
- cls acc: **88.9%**（8/9）
- **e2e: 5.5%**

## Detector FP → classifier 输出分布（`other` 拒绝情况）


| cls | count |
|-----|------:|
| `other` | 4 |
| `pl40` | 3 |

- **FP 被 `other` 拒绝：4/7（57%）**
- **有效 FP（漏到速度类）：3（0.003/img）**

明细：`/home/sicv/ws/tsr_speedlimit/runs/eval/e2e_self_v3/e2e_raw.json`

