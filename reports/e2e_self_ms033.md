# E2E 级联评估（detector → crop → classifier）

- 生成：2026-09-27T16:16:08.023815+00:00；split=test；det conf≥0.5；crop pad=20%；匹配 IoU≥0.5
- 域：det=`data/self/labels_det` mc=`data/self/labels_mc`；src=640x352（size640×1）；imgsz=[352, 640]
- 图像输入：`/home/sicv/ws/tsr_speedlimit/data/self/images/test`（GT 始终由 det/mc root 加载）
- detector：`runs/detector/tt_cropcolor1_ms033/weights/best.pt`
- classifier：`runs/classifier/v3/best.pt`（28 类，含 `other`）
- 图像 982；GT 185；检测框 56

## 总体

| 指标 | 值 |
|------|----:|
| Det recall@0.50 | 20.5%（38/185） |
| Det FP | 18（0.018/img） |
| Det FN | 147 |
| **Cls acc（det TP 且有 GT 类）** | **86.8%**（33/38） |
| **E2E acc = det recall × cls acc** | **17.8%** |

## 尺寸分桶（@640 等效 px）

| 桶 | GT | det recall | cls acc | e2e acc |
|----|---:|----------:|--------:|--------:|
| <8 | 0 | — | — | — |
| 8-12 | 24 | 0.0% | — | — |
| 12-16 | 62 | 1.6% | 100.0% | 1.6% |
| 16-24 | 62 | 17.7% | 72.7% | 12.9% |
| 24-32 | 30 | 63.3% | 89.5% | 56.7% |
| 32-48 | 4 | 100.0% | 100.0% | 100.0% |
| >=48 | 3 | 100.0% | 100.0% | 100.0% |

### 部署目标区间 [12, 28) px（GT 145）

- det recall: **16.6%**（24/145）
- cls acc: **83.3%**（20/24）
- **e2e: 13.8%**

## Detector FP → classifier 输出分布（`other` 拒绝情况）


| cls | count |
|-----|------:|
| `other` | 7 |
| `pl5` | 7 |
| `pl40` | 4 |

- **FP 被 `other` 拒绝：7/18（39%）**
- **有效 FP（漏到速度类）：11（0.011/img）**

明细：`/home/sicv/ws/tsr_speedlimit/runs/eval/e2e_self_ms033/e2e_raw.json`

