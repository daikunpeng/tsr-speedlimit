# TT100K-only 在线 640×352 位置裁切：检测器对照（tt_crop1）

## 实验设置

- 权重：`runs/detector/tt_crop1/weights/best.pt`；入口：`scripts/train_detector_crop352.py`；
  日志：`runs/tt_crop1_train.log`（100 epoch，约 48 分钟）。
- 与 smoke1 同一 `yolov5nu.pt` 初始化、TT100K train 6103 图 / 快速 val 203 图、
  100ep、imgsz=640、batch32、seed0、Ultralytics 默认增强。**只有 train** 在
  空间增强全部完成后（640×640、标注已对齐）裁为 640×352 窗口，再进入张量化：
  与部署画幅一致，且模拟自采牌位置分布——50% 概率把某个 GT 牌中心放入
  窗口顶部 5–30% 带（self 24–48px 牌 79% 在上缘 vs TT100K 8%），其余均匀偏移。
- 标注同步：框 y 坐标随窗口平移并裁剪到窗口界内，完全出窗的框与其 cls
  条目同步删除（与 `RandomPerspective.apply_instances` 相同的同步语义）；
  合成用例验证过平移/裁剪/丢弃/cls 对齐，close_mosaic 重建后变换仍在。
  源图与 TT100K test / self 评估输入一律不动；**未用任何 self 图像训练**。
- 训练期快速 val best mAP50≈0.746（smoke1≈0.728、tt_smear1≈0.739），
  结论以下列独立 test / self 为准。

## 结果对照（det conf=0.5，crop pad=20%，同一 cls v2）

| 指标 | smoke1 | tt_smear1 | **tt_crop1** | crop1−smoke1 |
|------|-------:|----------:|-------------:|-------------:|
| TT100K det recall@0.25 | 92.4% | 90.2% | 90.5% | −1.9pp |
| TT100K 12–28px recall@0.25 | 97.6% | 97.0% | 97.4% | −0.2pp |
| TT100K E2E@0.5 | 84.2% | 81.0% | 80.6% | −3.6pp |
| TT100K 12–28px E2E@0.5 | 93.7% | 92.7% | 91.1% | −2.6pp |
| self det recall@0.5 | 11.9%（22） | 14.6%（27） | **16.2%（30）** | **+4.3pp** |
| self det recall@0.25 | 18.4%（34） | 20.0%（37） | **21.6%（40）** | +3.2pp |
| self 24–32px recall@0.5 | 43.3% | 43.3% | **60.0%** | **+16.7pp** |
| self ≥48px recall@0.5 | 33.3%（1/3） | 100% | 100% | +66.7pp |
| self GT conf P50（0.0005 扫描） | 0.00165 | 0.00219 | **0.00363** | ×2.2 |
| self 无 IoU≥0.5 定位的 GT | 68/185 | 78/185 | **55/185** | −13 |
| self cls acc（det TP） | 81.8%（18/22） | 70.4% | **56.7%（17/30）** | −25.1pp |
| self E2E@0.5 | 9.7% | 10.3% | 9.2% | −0.5pp |
| self 事件通过 @conf0.5 | 2/3 | 1/3 | 2/3 | 持平 |
| self 事件通过 @conf0.25 | 3/3 | — | 2/3 | −1/3 |
| 峰值前 ≥24px 曾有一帧读对（辅助） | 4/9 | 5/9 | **6/9** | +2/9 |

- TT100K 掉点主因仍是 det TP 数（2152→2061）；两模型 det TP 上 cls acc 同为 97.4%。
- self cls 错误集中于 **pl40→other（6）、pl20→other（4）** 等暗红/低照度牌
  （conf0.25 时 16 错：pl40→other×8、pl20→other×5、pl40→pl5、pl40→pl15、pl60→other）。
- conf0.25 下 tt_crop1 唯一可验失败事件仍是 `_5` pl20（2/5 正确，多帧被判 `other`）；
  `_3` pl40 5/5、`_7` pl40 3/5 通过。未匹配速度值候选轨迹：1 条（`_3` pl5，最大仅 7.1px）。
- 事件总数 10、可验 3/10 的覆盖不变；辅助观察（近距曾读对）从 4/9 升至 6/9。

## 结论

1. **位置/画幅因素对 self 检测器是真实的**：同一 cls 下，tt_crop1 在关键
   24–32px 桶 +16.7pp、总体 +4.3pp、无定位 GT −13，是三轮中 self 域最好的
   检测器；≥48px 桶两个新模型均回到 100%（smoke1 1/3 为样本极小的桶）。
2. **瓶颈已移到分类器**：检测器新召回的牌大量是暗红/低照度外观，cls v2
   把它们判成 `other`（GT-crop 诊断：`_7` pl40、`_5` pl20 在 other 决策边界，
   1px 行偏移即翻转）。E2E 与事件通过率因此没有随 det 召回上升。
   即使检测完美，cls 当前上限也压着末段稳定读对。
3. TT100K 回归中等（E2E −3.6pp，主因训练时丢弃窗口外内容）；若后续以
   cls 颜色增广补上后半段，该回撤是否可接受需在替换默认权重时再权衡。

**当前不替换默认权重**：级联口径（E2E、事件）未超过 smoke1。
`tt_crop1` 记为 **self 域检测器候选**，待分类器颜色/曝光增广（见
`reports/frozen_diag.md` §8.1 证据；训练须另行确认）落地后合并决策。

## 下一步（均需用户确认后执行）

1. **分类器 v3（小成本，分钟级）**：只用 TT100K 现有 crop，在牌面局部做
   暗红圈/发白/亮度-伽马扰动增强重训 cls，复测 GT-crop 正确率与
   `_5`/`_7` 边界翻转，再跑 self E2E/事件。
2. 若 cls 修复后事件仍受检测限（无定位 55/185），再做**检测器侧**
   牌面局部颜色/曝光增广的一次全量训练（与本次位置裁切叠加或单因素另行设计）。
3. 新 trip 独立验收仍按 `docs/strategy.md` §8 延后执行。

明细：`reports/detector_eval_size_tt_crop1.md`、`reports/e2e_tt_crop1.md`、
`reports/detector_eval_size_self_crop1.md`、`reports/e2e_self_crop1.md`、
`reports/e2e_self_crop1_conf025.md`、`reports/event_eval_self_crop1{,_conf025}.md`、
`reports/self_score_sweep_crop1.md`；冻结权重诊断 `reports/frozen_diag.md`。
