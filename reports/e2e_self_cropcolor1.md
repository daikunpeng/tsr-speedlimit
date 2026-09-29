# E2E 级联评估（detector → crop → classifier）

- 生成：2026-09-24T09:49:09.976521+00:00；split=test；det conf≥0.5；crop pad=20%；匹配 IoU≥0.5
- 域：det=`data/self/labels_det` mc=`data/self/labels_mc`；src=640x352（size640×1）；imgsz=[352, 640]
- 图像输入：`/home/sicv/ws/tsr_speedlimit/data/self/images/test`（GT 始终由 det/mc root 加载）
- detector：`runs/detector/tt_cropcolor1/weights/best.pt`
- classifier：`runs/classifier/v3/best.pt`（28 类，含 `other`）
- 图像 982；GT 185；检测框 81

## 总体

| 指标 | 值 |
|------|----:|
| Det recall@0.50 | 37.3%（69/185） |
| Det FP | 12（0.012/img） |
| Det FN | 116 |
| **Cls acc（det TP 且有 GT 类）** | **91.3%**（63/69） |
| **E2E acc = det recall × cls acc** | **34.1%** |

## 尺寸分桶（@640 等效 px）

| 桶 | GT | det recall | cls acc | e2e acc |
|----|---:|----------:|--------:|--------:|
| <8 | 0 | — | — | — |
| 8-12 | 24 | 4.2% | 100.0% | 4.2% |
| 12-16 | 62 | 11.3% | 85.7% | 9.7% |
| 16-24 | 62 | 46.8% | 89.7% | 41.9% |
| 24-32 | 30 | 83.3% | 92.0% | 76.7% |
| 32-48 | 4 | 100.0% | 100.0% | 100.0% |
| >=48 | 3 | 100.0% | 100.0% | 100.0% |

### 部署目标区间 [12, 28) px（GT 145）

- det recall: **36.6%**（53/145）
- cls acc: **90.6%**（48/53）
- **e2e: 33.1%**

## Detector FP → classifier 输出分布（`other` 拒绝情况）


| cls | count |
|-----|------:|
| `other` | 8 |
| `pl40` | 4 |

- **FP 被 `other` 拒绝：8/12（67%）**
- **有效 FP（漏到速度类）：4（0.004/img）**

明细：`/home/sicv/ws/tsr_speedlimit/runs/eval/e2e_self_cropcolor1/e2e_raw.json`

