# E2E 级联评估（detector → crop → classifier）

- 生成：2026-09-24T01:25:13.824998+00:00；split=test；det conf≥0.25；crop pad=20%；匹配 IoU≥0.5
- 域：det=`data/self/labels_det` mc=`data/self/labels_mc`；src=640x352（size640×1）；imgsz=[352, 640]
- 图像输入：`/home/sicv/ws/tsr_speedlimit/data/self/images/test`（GT 始终由 det/mc root 加载）
- detector：`runs/detector/smoke1/weights/best.pt`
- classifier：`runs/classifier/v2/best.pt`（28 类，含 `other`）
- 图像 982；GT 185；检测框 58

## 总体

| 指标 | 值 |
|------|----:|
| Det recall@0.25 | 18.4%（34/185） |
| Det FP | 24（0.024/img） |
| Det FN | 151 |
| **Cls acc（det TP 且有 GT 类）** | **67.6%**（23/34） |
| **E2E acc = det recall × cls acc** | **12.4%** |

## 尺寸分桶（@640 等效 px）

| 桶 | GT | det recall | cls acc | e2e acc |
|----|---:|----------:|--------:|--------:|
| <8 | 0 | — | — | — |
| 8-12 | 24 | 12.5% | 0.0% | 0.0% |
| 12-16 | 62 | 3.2% | 50.0% | 1.6% |
| 16-24 | 62 | 9.7% | 83.3% | 8.1% |
| 24-32 | 30 | 56.7% | 70.6% | 40.0% |
| 32-48 | 4 | 100.0% | 75.0% | 75.0% |
| >=48 | 3 | 66.7% | 100.0% | 66.7% |

### 部署目标区间 [12, 28) px（GT 145）

- det recall: **12.4%**（18/145）
- cls acc: **66.7%**（12/18）
- **e2e: 8.3%**

## Detector FP → classifier 输出分布（`other` 拒绝情况）


| cls | count |
|-----|------:|
| `other` | 11 |
| `pl5` | 6 |
| `pl50` | 6 |
| `pl40` | 1 |

- **FP 被 `other` 拒绝：11/24（46%）**
- **有效 FP（漏到速度类）：13（0.013/img）**

明细：`/home/sicv/ws/tsr_speedlimit/runs/eval/e2e_self_smoke1_conf025/e2e_raw.json`

