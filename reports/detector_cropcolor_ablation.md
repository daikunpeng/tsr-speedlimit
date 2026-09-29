# TT100K-only 牌面颜色/曝光增广：检测器对照（tt_cropcolor1，B 步）

## 背景（A 步结论，`reports/diag_missed_frames.md`）

tt_crop1 的 self 大牌漏检 13 个，主导成因是**色彩/渲染偏移**而非尺寸或锐度：
漏检组红度 P50=6、亮度 P50=90 vs 检出组 30/184；分三类——暗/褪色红牌（~8/13，
`_9`/`_1`/`_7`）、过曝发白（`_4` 帧111）、非颜色（`_3` 帧67 等 ~3 帧需目检）。

## 实验设置

- 权重：`runs/detector/tt_cropcolor1/weights/best.pt`；入口 `scripts/train_detector_crop_color.py`；
  日志 `runs/tt_cropcolor1_train.log`（100 ep，约 48 分钟）。
- 在 tt_crop1 配方（在线 640×352 位置裁切）之上，训练期新增 **PlateColorAug**：
  对每个 GT 牌框以概率 0.5 施加 暗红/发白/亮度-饱和-伽马 抖动（与 cls v3 同三种
  模式，BGR 版本；hue 不动），径向羽化融回，**框几何不变、无标注重同步**。
  val/test 不增强；不用任何 self 图像。训练期快速 val best mAP50≈0.642
  （crop1≈0.746；掉点来自干净 val 牌被增广权衡，非结论依据）。
- 分类器固定 cls v3；对照基线 = tt_crop1 + v3。

## 结果（det conf=0.5，crop pad=20%）

| 指标 | smoke1+v2 | smoke1+v3 | crop1+v3 | **cropcolor1+v3** |
|------|----------:|----------:|---------:|------------------:|
| TT100K det recall@0.5 | 86.4% | 86.4% | 82.8% | 77.2% |
| TT100K det recall@0.25 | 92.4% | 92.4% | 90.5% | 86.4% |
| TT100K 12–28px recall@0.5 | 96.2% | 96.2% | 93.4% | 93.4% |
| TT100K E2E@0.5 | 84.2% | 84.4% | 80.4% | 75.5% |
| TT100K 12–28px E2E@0.5 | 93.7% | 93.6% | 90.6% | 90.9% |
| self det recall@0.5 | 11.9% | 11.9% | 16.2% | **37.3%（69/185）** |
| self 24–32px @0.5 | 43.3% | 43.3% | 60.0% | **83.3%** |
| self 16–24px @0.5 | 3.2% | 3.2% | 6.5% | **46.8%** |
| self 12–16px @0.5 | 0% | 0% | 0% | 11.3% |
| self 无定位 GT（IoU<0.5） | 68 | 68 | 55 | **11** |
| self GT conf P50（0.0005 扫描） | 0.00165 | 0.00165 | 0.00363 | **0.179** |
| self cls acc（det TP） | 81.8% | 90.9% | 93.3% | 91.3%（63/69） |
| self E2E@0.5 | 9.7% | 10.8% | 15.1% | **34.1%** |
| self 事件通过 @0.5 | 2/3 | 3/3 | 3/3 | **3/3** |
| 峰值前 ≥24px 曾有一帧读对（辅助） | 4/9 | 4/9 | 6/9 | **8/9** |
| self FP/img @0.5（漏到速度类） | 0.007(3) | 0.007(3) | 0.010(6) | 0.012(4) |

## 解读

1. **牌面颜色增广是目标域最大单一杠杆**：self det recall@0.5 +21pp（vs crop1）、
   +25pp（vs smoke1）；无定位 55→11；GT 分数中位 ×100。级联 E2E 9.7%→34.1%（×3.5）。
   关键桶 24–32px 83%、16–24px 47%，末段事件 3/3、近距曾读对 8/9。
2. **代价是 TT100K 干净域回退**：E2E 84.2→75.5%（−8.7pp），主因 <8px 桶
   （64→46%）与 ≥48px（85→77%）的干净牌置信度下降——一半训练牌被颜色扰动后，
   干净极端尺寸牌变不自信。12–28px 目标带仍 93.4% recall、90.9% E2E，回退有限。
3. 业务目标是 GS130WI 临近稳定读对，TT100K 只是代理训练源/回归参照；
   该权衡方向正确，但 −8.7pp 偏大，值得用更温和的增广强度再探一次。

## 建议与待决

- 默认组合候选：**tt_cropcolor1 + cls v3**（self 最优）。
- 可选后续（各需一次全量训练，另行授权）：
  1. 把 PlateColorAug 概率 0.5→0.25/0.3 或收窄强度，尝试找回部分 TT100K 干净牌；
  2. 若要保持 TT100K 近 84%：把颜色增广限制在 12–48px 目标带（<8px/≥48px 不动）。
- 仍需新 trip 独立验收（当前 3/10 事件覆盖、6 trip 已参与方案选择）。

明细：`reports/diag_missed_frames.md`、`reports/detector_eval_size_self_cropcolor1.md`、
`reports/e2e_self_cropcolor1.md`、`reports/event_eval_self_cropcolor1.md`、
`reports/detector_eval_size_tt_cropcolor1.md`、`reports/e2e_tt_cropcolor1.md`、
`reports/self_score_sweep_cropcolor1.md`；漏检帧图 `runs/vis/missed_frames/`。
