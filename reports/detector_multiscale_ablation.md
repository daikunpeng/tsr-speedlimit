# 多尺度训练消融：Ultralytics 内置 `multi_scale` 能否替代 TTA？

## 动机

TTA（推理时 1.0/0.83/0.67 三尺度 + 翻转，NMS 合并）已验证能修复 `_1` 的尺度混叠
（连续 3 帧→pass），但推理 ~3×。本实验尝试把「多尺度」在**训练阶段**教给模型
（Ultralytics 内置 `multi_scale`，`imgsz·(1±ms)` 随机缩放每个 train batch），
期望推理时单尺度就能拿到同等鲁棒性，省掉 3× 算力。

## 设置

- 配方 = `tt_cropcolor1`（crop352 + PlateColorAug，TT100K-only，100ep，seed 0，batch 32）
  **唯一变量**：`multi_scale=0.33`（尺度范围 sf ∈ [0.65, 1.35]，覆盖 TTA 的 0.67/0.83 下采样）。
- 训练显存峰值 5.7G，无 OOM；best mAP50≈0.63（cropcolor1 ≈0.64@88ep）。
- 权重 `runs/detector/tt_cropcolor1_ms033/weights/best.pt`；评估全部**单尺度、无 TTA**。

## 结果（det conf=0.5）

| 指标 | cropcolor1（固定尺度） | ms033（multi_scale 0.33） | 差值 |
|------|-------:|----:|-----:|
| self det recall@0.5 | **37.3%** | 20.5% | **−16.8pp** |
| self 12–16px @0.5 | ~12.9% | 1.6% | −11.3pp |
| self 16–24px @0.5 | 46.8% | 17.7% | −29.1pp |
| self E2E@0.5 | **34.1%** | 17.8% | −16.3pp |
| **self 事件通过（新标准）** | **7/9** | **3/9** | **−4** |
| `_1` pl60 最长连续正确 | 2（帧 44–45） | 1（帧 51） | 变差 |
| TT100K det recall@0.5 | 77.2% | 76.1% | −1.1pp |
| TT100K E2E@0.5 | 75.5% | 74.1% | −1.4pp |

## 关键结论（负结果）

1. **尺度塌陷没有被消除**：`diag_scale_sweep` 显示 native 1.0× 下 `_1` 的 15–20px 区间
   conf 仍接近 0（帧 46=0.011、47=0.000、48=0.0007、49=0.002、50=0.002），
   帧 52 仍 0.169；`_1` 事件最长连续正确反而从 2 降到 1。
2. **整图 multi_scale 训练显著损害 self 域召回**：det@0.5 37.3→20.5%，尤其小尺寸
   （12–16px 12.9→1.6%、16–24px 46.8→17.7%）；事件 7/9→3/9。
3. TT100K 基本中性（略降 1.4pp），说明「多尺度」本身不带来收益，只是在稀释。

## 机制解释（为什么失败）

- **训练约一半时间向下缩放**（sf 0.65–1.0）：把 12–28px 的小牌压到 <10px 亚可学习区，
  模型反而学会「小牌不可信」，native 尺度下小牌置信被稀释。
- **Ultralytics 本就有 mosaic + RandomPerspective 的 scale jitter（0.5–1.5）**：再叠加
  整图 multi_scale 属于冗余，且对已调好的 crop352+颜色增广配方是净干扰。
- **TTA 与多尺度训练机制根本不同**：TTA 是多尺度推理后 **NMS 选优**（总能挑到最干净的
  尺度），而训练期 multi_scale 是「跨尺度平均化」，二者不可互相替代。

## 结论与建议

- **放弃整图 `multi_scale` 训练**；它不能替代 TTA，且反向损害 cropcolor1 配方。
- `_1` 尺度塌陷的正确部署侧解法回到两条路（见 `docs/strategy.md` §8.5）：
  1. **时序跟踪 + 多帧投票（首选，更省算力）**：`_1` 每帧都被定位（IoU 0.89–0.97），
     只需在高分帧（51/53）读出、跨过塌陷帧（46–50）即可。
  2. **TTA**（已验证有效，代价 3× 算力），作为离线复核或高算力场景兜底。
- 本实验为已授权的一次全量训练对照，结果已用；ms033 权重仅存档，不替换任何默认。

明细：`reports/detector_eval_size_self_ms033.md`、`reports/e2e_self_ms033.md`、
`reports/event_eval_self_ms033.md`、`reports/detector_eval_size_tt_ms033.md`、
`reports/e2e_tt_ms033.md`。
