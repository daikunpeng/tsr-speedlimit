# E2E 级联评估（detector → crop → classifier）

- 生成：2026-09-24T08:03:14.057962+00:00；split=test；det conf≥0.5；crop pad=20%；匹配 IoU≥0.5
- 域：det=`data/self/labels_det` mc=`data/self/labels_mc`；src=640x352（size640×1）；imgsz=[352, 640]
- 图像输入：`/home/sicv/ws/tsr_speedlimit/data/self/images/test`（GT 始终由 det/mc root 加载）
- detector：`runs/detector/tt_crop1/weights/best.pt`
- classifier：`runs/classifier/v3/best.pt`（28 类，含 `other`）
- 图像 982；GT 185；检测框 40

## 总体

| 指标 | 值 |
|------|----:|
| Det recall@0.50 | 16.2%（30/185） |
| Det FP | 10（0.010/img） |
| Det FN | 155 |
| **Cls acc（det TP 且有 GT 类）** | **93.3%**（28/30） |
| **E2E acc = det recall × cls acc** | **15.1%** |

## 尺寸分桶（@640 等效 px）

| 桶 | GT | det recall | cls acc | e2e acc |
|----|---:|----------:|--------:|--------:|
| <8 | 0 | — | — | — |
| 8-12 | 24 | 8.3% | 100.0% | 8.3% |
| 12-16 | 62 | 0.0% | — | — |
| 16-24 | 62 | 6.5% | 100.0% | 6.5% |
| 24-32 | 30 | 60.0% | 88.9% | 53.3% |
| 32-48 | 4 | 75.0% | 100.0% | 75.0% |
| >=48 | 3 | 100.0% | 100.0% | 100.0% |

### 部署目标区间 [12, 28) px（GT 145）

- det recall: **9.7%**（14/145）
- cls acc: **92.9%**（13/14）
- **e2e: 9.0%**

## Detector FP → classifier 输出分布（`other` 拒绝情况）


| cls | count |
|-----|------:|
| `pl40` | 6 |
| `other` | 4 |

- **FP 被 `other` 拒绝：4/10（40%）**
- **有效 FP（漏到速度类）：6（0.006/img）**

明细：`/home/sicv/ws/tsr_speedlimit/runs/eval/e2e_self_crop1v3/e2e_raw.json`

