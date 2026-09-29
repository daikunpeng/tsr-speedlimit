# E2E 级联评估（detector → crop → classifier）

- 生成：2026-09-24T08:03:31.469973+00:00；split=test；det conf≥0.25；crop pad=20%；匹配 IoU≥0.5
- 域：det=`data/self/labels_det` mc=`data/self/labels_mc`；src=640x352（size640×1）；imgsz=[352, 640]
- 图像输入：`/home/sicv/ws/tsr_speedlimit/data/self/images/test`（GT 始终由 det/mc root 加载）
- detector：`runs/detector/tt_crop1/weights/best.pt`
- classifier：`runs/classifier/v3/best.pt`（28 类，含 `other`）
- 图像 982；GT 185；检测框 71

## 总体

| 指标 | 值 |
|------|----:|
| Det recall@0.25 | 21.6%（40/185） |
| Det FP | 31（0.032/img） |
| Det FN | 145 |
| **Cls acc（det TP 且有 GT 类）** | **92.5%**（37/40） |
| **E2E acc = det recall × cls acc** | **20.0%** |

## 尺寸分桶（@640 等效 px）

| 桶 | GT | det recall | cls acc | e2e acc |
|----|---:|----------:|--------:|--------:|
| <8 | 0 | — | — | — |
| 8-12 | 24 | 8.3% | 100.0% | 8.3% |
| 12-16 | 62 | 1.6% | 100.0% | 1.6% |
| 16-24 | 62 | 17.7% | 90.9% | 16.1% |
| 24-32 | 30 | 63.3% | 89.5% | 56.7% |
| 32-48 | 4 | 100.0% | 100.0% | 100.0% |
| >=48 | 3 | 100.0% | 100.0% | 100.0% |

### 部署目标区间 [12, 28) px（GT 145）

- det recall: **15.2%**（22/145）
- cls acc: **90.9%**（20/22）
- **e2e: 13.8%**

## Detector FP → classifier 输出分布（`other` 拒绝情况）


| cls | count |
|-----|------:|
| `pl40` | 15 |
| `other` | 14 |
| `pl5` | 2 |

- **FP 被 `other` 拒绝：14/31（45%）**
- **有效 FP（漏到速度类）：17（0.017/img）**

明细：`/home/sicv/ws/tsr_speedlimit/runs/eval/e2e_self_crop1v3_c025/e2e_raw.json`

