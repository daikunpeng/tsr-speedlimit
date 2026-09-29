# E2E 级联评估（detector → crop → classifier）

- 生成：2026-09-24T05:16:25.990070+00:00；split=test；det conf≥0.25；crop pad=20%；匹配 IoU≥0.5
- 域：det=`data/self/labels_det` mc=`data/self/labels_mc`；src=640x352（size640×1）；imgsz=[352, 640]
- 图像输入：`/home/sicv/ws/tsr_speedlimit/data/self/images/test`（GT 始终由 det/mc root 加载）
- detector：`runs/detector/tt_crop1/weights/best.pt`
- classifier：`runs/classifier/v2/best.pt`（28 类，含 `other`）
- 图像 982；GT 185；检测框 71

## 总体

| 指标 | 值 |
|------|----:|
| Det recall@0.25 | 21.6%（40/185） |
| Det FP | 31（0.032/img） |
| Det FN | 145 |
| **Cls acc（det TP 且有 GT 类）** | **60.0%**（24/40） |
| **E2E acc = det recall × cls acc** | **13.0%** |

## 尺寸分桶（@640 等效 px）

| 桶 | GT | det recall | cls acc | e2e acc |
|----|---:|----------:|--------:|--------:|
| <8 | 0 | — | — | — |
| 8-12 | 24 | 8.3% | 0.0% | 0.0% |
| 12-16 | 62 | 1.6% | 100.0% | 1.6% |
| 16-24 | 62 | 17.7% | 63.6% | 11.3% |
| 24-32 | 30 | 63.3% | 63.2% | 40.0% |
| 32-48 | 4 | 100.0% | 50.0% | 50.0% |
| >=48 | 3 | 100.0% | 66.7% | 66.7% |

### 部署目标区间 [12, 28) px（GT 145）

- det recall: **15.2%**（22/145）
- cls acc: **59.1%**（13/22）
- **e2e: 9.0%**

## Detector FP → classifier 输出分布（`other` 拒绝情况）


| cls | count |
|-----|------:|
| `pl5` | 14 |
| `other` | 7 |
| `pl50` | 6 |
| `pl40` | 4 |

- **FP 被 `other` 拒绝：7/31（23%）**
- **有效 FP（漏到速度类）：24（0.024/img）**

明细：`/home/sicv/ws/tsr_speedlimit/runs/eval/e2e_self_crop1_conf025/e2e_raw.json`

