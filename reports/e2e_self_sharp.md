# E2E 级联评估（detector → crop → classifier）

- 生成：2026-09-23T14:33:57.780998+00:00；split=test；det conf≥0.5；crop pad=20%；匹配 IoU≥0.5
- 域：det=`data/self/labels_det` mc=`data/self/labels_mc`；src=640x352（size640×1）；imgsz=[352, 640]
- 图像输入：`/home/sicv/ws/tsr_speedlimit/runs/preproc/sharp/images/test`（GT 始终由 det/mc root 加载）
- detector：`runs/detector/smoke1/weights/best.pt`
- classifier：`runs/classifier/v2/best.pt`（28 类，含 `other`）
- 图像 982；GT 185；检测框 26

## 总体

| 指标 | 值 |
|------|----:|
| Det recall@0.50 | 11.4%（21/185） |
| Det FP | 5（0.005/img） |
| Det FN | 164 |
| **Cls acc（det TP 且有 GT 类）** | **66.7%**（14/21） |
| **E2E acc = det recall × cls acc** | **7.6%** |

## 尺寸分桶（@640 等效 px）

| 桶 | GT | det recall | cls acc | e2e acc |
|----|---:|----------:|--------:|--------:|
| <8 | 0 | — | — | — |
| 8-12 | 24 | 4.2% | 0.0% | 0.0% |
| 12-16 | 62 | 0.0% | — | — |
| 16-24 | 62 | 3.2% | 100.0% | 3.2% |
| 24-32 | 30 | 43.3% | 69.2% | 30.0% |
| 32-48 | 4 | 100.0% | 75.0% | 75.0% |
| >=48 | 3 | 33.3% | 0.0% | 0.0% |

### 部署目标区间 [12, 28) px（GT 145）

- det recall: **6.2%**（9/145）
- cls acc: **88.9%**（8/9）
- **e2e: 5.5%**

## Detector FP → classifier 输出分布（`other` 拒绝情况）


| cls | count |
|-----|------:|
| `other` | 3 |
| `pl50` | 2 |

- **FP 被 `other` 拒绝：3/5（60%）**
- **有效 FP（漏到速度类）：2（0.002/img）**

明细：`/home/sicv/ws/tsr_speedlimit/runs/eval/e2e_self_sharp/e2e_raw.json`

