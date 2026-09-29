# E2E 级联评估（detector → crop → classifier）

- 生成：2026-09-24T11:10:08.845610+00:00；split=test；det conf≥0.5；crop pad=20%；匹配 IoU≥0.5
- 域：det=`data/self/labels_det` mc=`data/self/labels_mc`；src=640x352（size640×1）；imgsz=[352, 640]
- 图像输入：`/home/sicv/ws/tsr_speedlimit/data/self/images/test`（GT 始终由 det/mc root 加载）
- detector：`runs/detector/tt_cropcolor1/weights/best.pt`
- classifier：`runs/classifier/v3/best.pt`（28 类，含 `other`）
- 图像 982；GT 185；检测框 111

## 总体

| 指标 | 值 |
|------|----:|
| Det recall@0.50 | 42.2%（78/185） |
| Det FP | 33（0.034/img） |
| Det FN | 107 |
| **Cls acc（det TP 且有 GT 类）** | **92.3%**（72/78） |
| **E2E acc = det recall × cls acc** | **38.9%** |

## 尺寸分桶（@640 等效 px）

| 桶 | GT | det recall | cls acc | e2e acc |
|----|---:|----------:|--------:|--------:|
| <8 | 0 | — | — | — |
| 8-12 | 24 | 4.2% | 100.0% | 4.2% |
| 12-16 | 62 | 12.9% | 87.5% | 11.3% |
| 16-24 | 62 | 54.8% | 91.2% | 50.0% |
| 24-32 | 30 | 93.3% | 92.9% | 86.7% |
| 32-48 | 4 | 100.0% | 100.0% | 100.0% |
| >=48 | 3 | 100.0% | 100.0% | 100.0% |

### 部署目标区间 [12, 28) px（GT 145）

- det recall: **42.1%**（61/145）
- cls acc: **91.8%**（56/61）
- **e2e: 38.6%**

## Detector FP → classifier 输出分布（`other` 拒绝情况）


| cls | count |
|-----|------:|
| `other` | 23 |
| `pl40` | 6 |
| `pl5` | 3 |
| `pl60` | 1 |

- **FP 被 `other` 拒绝：23/33（70%）**
- **有效 FP（漏到速度类）：10（0.010/img）**

明细：`/home/sicv/ws/tsr_speedlimit/runs/eval/e2e_self_cropcolor1_tta/e2e_raw.json`

