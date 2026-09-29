# E2E 级联评估（detector → crop → classifier）

- 生成：2026-09-24T05:11:06.051489+00:00；split=test；det conf≥0.5；crop pad=20%；匹配 IoU≥0.5
- 域：det=`data/self/labels_det` mc=`data/self/labels_mc`；src=640x352（size640×1）；imgsz=[352, 640]
- 图像输入：`/home/sicv/ws/tsr_speedlimit/data/self/images/test`（GT 始终由 det/mc root 加载）
- detector：`runs/detector/tt_crop1/weights/best.pt`
- classifier：`runs/classifier/v2/best.pt`（28 类，含 `other`）
- 图像 982；GT 185；检测框 40

## 总体

| 指标 | 值 |
|------|----:|
| Det recall@0.50 | 16.2%（30/185） |
| Det FP | 10（0.010/img） |
| Det FN | 155 |
| **Cls acc（det TP 且有 GT 类）** | **56.7%**（17/30） |
| **E2E acc = det recall × cls acc** | **9.2%** |

## 尺寸分桶（@640 等效 px）

| 桶 | GT | det recall | cls acc | e2e acc |
|----|---:|----------:|--------:|--------:|
| <8 | 0 | — | — | — |
| 8-12 | 24 | 8.3% | 0.0% | 0.0% |
| 12-16 | 62 | 0.0% | — | — |
| 16-24 | 62 | 6.5% | 50.0% | 3.2% |
| 24-32 | 30 | 60.0% | 66.7% | 40.0% |
| 32-48 | 4 | 75.0% | 33.3% | 25.0% |
| >=48 | 3 | 100.0% | 66.7% | 66.7% |

### 部署目标区间 [12, 28) px（GT 145）

- det recall: **9.7%**（14/145）
- cls acc: **50.0%**（7/14）
- **e2e: 4.8%**

## Detector FP → classifier 输出分布（`other` 拒绝情况）


| cls | count |
|-----|------:|
| `pl5` | 5 |
| `pl50` | 2 |
| `other` | 2 |
| `pl40` | 1 |

- **FP 被 `other` 拒绝：2/10（20%）**
- **有效 FP（漏到速度类）：8（0.008/img）**

明细：`/home/sicv/ws/tsr_speedlimit/runs/eval/e2e_self_crop1/e2e_raw.json`

