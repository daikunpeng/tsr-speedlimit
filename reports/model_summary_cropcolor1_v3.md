# 模型总结：tt_cropcolor1（检测器）+ cls v3（分类器）

> 当前**目标域（GS130WI 自采）最优**的两级流水线组合。
> 本文只讲这一组合：它由什么构成、怎么训练、性能如何、代价与局限、下一步。

---

## 1. 组合是什么

| 阶段 | 模型 | 权重 | 任务 |
|------|------|------|------|
| 检测 | YOLOv5nu（`tt_cropcolor1`） | `runs/detector/tt_cropcolor1/weights/best.pt` | 单类 `speed_limit_sign` 定位 |
| 分类 | 245k 参数 tiny classifier（`v3`） | `runs/classifier/v3/best.pt` | 28 类：27 个限速值 + `other`（拒收） |

流水线：`图 → 检测器（框出限速牌）→ crop + 20% pad → 分类器（读数值）`。
输入 640×352（GS130WI）；评估 det conf≥0.5、crop IoU≥0.5。

---

## 2. 检测器 tt_cropcolor1

**训练入口** `scripts/train_detector_crop_color.py`，TT100K-only、100ep、seed 0、batch 32、
imgsz 640（训练期在线裁切 640×352），val/test 不增强。

相比 smoke1 增加了**两个训练期增强**（针对 self 域根因，见 `reports/detector_cropcolor_ablation.md`）：

1. **CropWide352**：每图随机取 640×352 窗口，50% 概率把 GT 牌放到窗口顶部带
   （对齐部署画幅与 self 牌多在上缘的分布）；框几何同步平移/裁剪/删除。
2. **PlateColorAug**（概率 0.5）：对每个 GT 牌框局部施加 暗红 / 发白 / 亮度-饱和-伽马
   抖动（hue 不动，径向羽化），针对 `_4` 过曝、`_9` 暗红等色彩偏移。

关键收益（vs smoke1）：self 无定位 68→11、GT conf P50 0.00165→0.179、det@0.5 11.9%→37.3%。

---

## 3. 分类器 v3

**训练入口** `scripts/train_classifier.py --color-aug 0.5`，输入 64×64、245k 参数、
scales [1.0,1.3,1.5,2.0]，30ep。数据 = v2 同源（正样本 30936 crop + 负样本 `data/cls_neg/`），
**唯一新增**训练期牌面颜色/曝光增强（概率 0.5，hue 不动，test 不增强）。

| 指标 | 值 |
|------|----:|
| test 总 acc（含 other） | 95.64% |
| speed 类 acc | 95.69% |
| 负样本拒绝率（→other） | 95.3%（label 98.9% / detfp 58.1%） |

self 域关键收益（暗红/过曝牌）：GT-crop 分类 112→**159/185**；`_7` 暗红 pl40 8→19/20、
`_5` pl20 15→23/24、`other` 误杀 51→14。

---

## 4. 性能总表（det conf=0.5，20% pad，无 TTA）

| 指标 | self（982 图/185 GT） | TT100K（3067 图/2490 GT） |
|------|-----:|-----:|
| det recall@0.5 | **37.3%**（69/185） | 77.2% |
| det recall@0.25 | 47.0% | 86.4% |
| 12–28px recall@0.5 | 36.6% | 93.4% |
| cls acc（det TP） | 91.3%（63/69） | — |
| **E2E@0.5** | **34.1%** | **75.5%** |
| 12–28px E2E@0.5 | — | 90.9% |
| det FP（漏到速度类） | 12（0.004/img） | 0.047/img |

### self 尺寸分桶（@640 等效 px）

| 桶 | GT | recall@0.5 |
|----|---:|---:|
| 8–12 | 24 | 4.2% |
| 12–16 | 62 | 11.3% |
| 16–24 | 62 | 46.8% |
| 24–32 | 30 | **83.3%** |
| 32–48 | 4 | 100% |
| ≥48 | 3 | 100% |

### self 事件通过（新标准：生命周期内连续 3 帧 E2E 正确）

**7/9 通过**（2 失败：`_4` pl30 过曝连续 2 帧+1 错值；`_1` pl60 尺度塌陷连续 2 帧；
1 不可验：`_9` pl40 生命周期仅 1 帧）。近距曾读对辅助 8/9。

---

## 5. TTA 增强变体（可选，推理时开启）

`eval_detector_size.py` / `eval_e2e.py` 加 `--tta`（`predict(augment=True)`，尺度
1.0/0.83/0.67 + 翻转后 NMS 合并），**零训练**、无需换权重：

| 指标 | 无 TTA | + TTA |
|------|-----:|-----:|
| self det@0.5 | 37.3% | **42.2%** |
| self E2E | 34.1% | **38.9%** |
| self 事件通过 | 7/9 | **8/9**（`_1` 连续 3 帧→pass） |
| TT100K E2E | 75.5% | **78.2%** |
| self FP/img | 0.012 | 0.034 |
| TT100K FP/img | 0.047 | 0.074 |

TTA 精准修复 `_1` 的尺度混叠，但推理 ~3×、FP 上升，记为部署侧候选（离线复核/高算力兜底）。

---

## 6. 权衡与已知局限

1. **TT100K 干净域回退**：E2E 84.2%（smoke1）→75.5%，主因 <8px 与 ≥48px 干净牌置信下降；
   12–28px 目标带仍 93.4% recall / 90.9% E2E，回退有限。这是「为新相机买单」的方向性取舍。
2. **小尺寸仍弱**：self 8–16px 基本检不出（这是部署早期、距离尚远阶段，业务可接受）。
3. **剩余未解**：`_4` 过曝、`_1` 尺度混叠（可 TTA 兜底）、`_9` 生命周期过短。
4. **尚未正式替换默认权重**：本组合为候选；`data/self` 仍是纯测试集，任何训练主路径不得依赖它。

---

## 7. 下一步方向

- **时序跟踪 + 多帧投票**（首选）：`_1` 每帧都定位（IoU 0.89–0.97），跨过低分帧在高分帧读出，省算力。
- **实验 A 恢复**：颜色增广收窄强度 / 限定 12–48px 目标带，找回部分 TT100K 干净牌。
- **TTA**：已可用于兜底。
- 新 trip 独立验收（当前仅 3/10 事件可覆盖，6 trip 已参与方案选择）。

---

## 8. 文件索引

- 检测器训练：`scripts/train_detector_crop_color.py`；权重 `runs/detector/tt_cropcolor1/weights/best.pt`
- 分类器训练：`scripts/train_classifier.py`（`--color-aug`）；权重 `runs/classifier/v3/best.pt` + `class_map.json`
- 类别定义：`data/self/classes.txt`（82 类 + `check_me`）；分类器 28 类见 `class_map.json`
- 评估：`scripts/eval_detector_size.py`、`scripts/eval_e2e.py`（含 `--tta`）、`scripts/eval_self_events.py`
- 报告：`detector_cropcolor_ablation.md`、`classifier_v3.md`、`cls_v3_cascade.md`、
  `e2e_self_cropcolor1.md`、`e2e_tt_cropcolor1.md`、`event_eval_self_cropcolor1.md`、
  `tta_probe.md`、`detector_multiscale_ablation.md`
