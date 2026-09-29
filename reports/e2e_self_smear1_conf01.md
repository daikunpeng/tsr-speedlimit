# E2E 级联评估（detector → crop → classifier）

- 生成：2026-09-24T01:20:31.821906+00:00；split=test；det conf≥0.1；crop pad=20%；匹配 IoU≥0.5
- 域：det=`data/self/labels_det` mc=`data/self/labels_mc`；src=640x352（size640×1）；imgsz=[352, 640]
- 图像输入：`/home/sicv/ws/tsr_speedlimit/data/self/images/test`（GT 始终由 det/mc root 加载）
- detector：`runs/detector/tt_smear1/weights/best.pt`
- classifier：`runs/classifier/v2/best.pt`（28 类，含 `other`）
- 图像 982；GT 185；检测框 82

## 总体

| 指标 | 值 |
|------|----:|
| Det recall@0.10 | 25.4%（47/185） |
| Det FP | 35（0.036/img） |
| Det FN | 138 |
| **Cls acc（det TP 且有 GT 类）** | **57.4%**（27/47） |
| **E2E acc = det recall × cls acc** | **14.6%** |

## 尺寸分桶（@640 等效 px）

| 桶 | GT | det recall | cls acc | e2e acc |
|----|---:|----------:|--------:|--------:|
| <8 | 0 | — | — | — |
| 8-12 | 24 | 16.7% | 0.0% | 0.0% |
| 12-16 | 62 | 4.8% | 33.3% | 1.6% |
| 16-24 | 62 | 21.0% | 61.5% | 12.9% |
| 24-32 | 30 | 66.7% | 70.0% | 46.7% |
| 32-48 | 4 | 100.0% | 50.0% | 50.0% |
| >=48 | 3 | 100.0% | 66.7% | 66.7% |

### 部署目标区间 [12, 28) px（GT 145）

- det recall: **18.6%**（27/145）
- cls acc: **55.6%**（15/27）
- **e2e: 10.3%**

## Detector FP → classifier 输出分布（`other` 拒绝情况）


| cls | count |
|-----|------:|
| `pl5` | 20 |
| `pl50` | 8 |
| `other` | 6 |
| `pl40` | 1 |

- **FP 被 `other` 拒绝：6/35（17%）**
- **有效 FP（漏到速度类）：29（0.030/img）**

明细：`/home/sicv/ws/tsr_speedlimit/runs/eval/e2e_self_smear1_conf01/e2e_raw.json`

