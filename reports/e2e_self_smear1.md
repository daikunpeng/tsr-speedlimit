# E2E 级联评估（detector → crop → classifier）

- 生成：2026-09-24T01:14:08.220683+00:00；split=test；det conf≥0.5；crop pad=20%；匹配 IoU≥0.5
- 域：det=`data/self/labels_det` mc=`data/self/labels_mc`；src=640x352（size640×1）；imgsz=[352, 640]
- 图像输入：`/home/sicv/ws/tsr_speedlimit/data/self/images/test`（GT 始终由 det/mc root 加载）
- detector：`runs/detector/tt_smear1/weights/best.pt`
- classifier：`runs/classifier/v2/best.pt`（28 类，含 `other`）
- 图像 982；GT 185；检测框 29

## 总体

| 指标 | 值 |
|------|----:|
| Det recall@0.50 | 14.6%（27/185） |
| Det FP | 2（0.002/img） |
| Det FN | 158 |
| **Cls acc（det TP 且有 GT 类）** | **70.4%**（19/27） |
| **E2E acc = det recall × cls acc** | **10.3%** |

## 尺寸分桶（@640 等效 px）

| 桶 | GT | det recall | cls acc | e2e acc |
|----|---:|----------:|--------:|--------:|
| <8 | 0 | — | — | — |
| 8-12 | 24 | 8.3% | 0.0% | 0.0% |
| 12-16 | 62 | 0.0% | — | — |
| 16-24 | 62 | 9.7% | 83.3% | 8.1% |
| 24-32 | 30 | 43.3% | 84.6% | 36.7% |
| 32-48 | 4 | 75.0% | 33.3% | 25.0% |
| >=48 | 3 | 100.0% | 66.7% | 66.7% |

### 部署目标区间 [12, 28) px（GT 145）

- det recall: **8.3%**（12/145）
- cls acc: **75.0%**（9/12）
- **e2e: 6.2%**

## Detector FP → classifier 输出分布（`other` 拒绝情况）


| cls | count |
|-----|------:|
| `pl50` | 1 |
| `pl5` | 1 |

明细：`/home/sicv/ws/tsr_speedlimit/runs/eval/e2e_self_smear1/e2e_raw.json`

