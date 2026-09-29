# TTA 零训练探针：多尺度推理能否缓解“尺度混叠”

## 设置

- 冻结 `tt_cropcolor1` 检测器 + `cls v3`，**不训练**。评估脚本加 `--tta` 后传
  `predict(augment=True)`：Ultralytics 内置 TTA = 尺度 [1.0, 0.83, 0.67] + 水平翻转，
  经 NMS 合并（`nn/tasks.py::_predict_augment`，只向下缩放）。
- 新验收标准（`docs/strategy.md` §8.5）：生命周期内**连续 3 帧 E2E 正确**即通过；
  连续 ≥3 同错值不通过；<3 帧不可验。

## 结果（cropcolor1 + cls v3，det conf=0.5）

| 指标 | 无 TTA | TTA | 差值 |
|------|-------:|----:|-----:|
| self det recall@0.5 | 37.3%（69） | **42.2%（78）** | +4.9pp |
| self 24–32px @0.5 | 83.3% | **93.3%** | +10.0pp |
| self 16–24px @0.5 | 46.8% | **54.8%** | +8.0pp |
| self E2E@0.5 | 34.1% | **38.9%** | +4.8pp |
| self FP/img @0.5（漏到速度类） | 0.012（4） | 0.034（10） | ↑ |
| **self 事件通过（新标准）** | **7/9** | **8/9** | **+1** |
| TT100K det recall@0.5 | 77.2% | 80.0% | +2.8pp |
| TT100K E2E@0.5 | 75.5% | 78.2% | +2.7pp |
| TT100K 12–28px E2E | 90.9% | 93.2% | +2.3pp |
| TT100K FP/img @0.5 | 0.047（145） | 0.074（226） | ↑ |

## 关键结论

1. **TTA 精准解决了 `_1` 的尺度混叠**：`_1` pl60 从“最长连续正确 2 帧”变成
   **连续 3 帧（51–53，22–28px）→ pass**。用户判断正确——下采样恰好让帧 51/52
   进入高分带。事件通过从 **7/9 → 8/9**。
2. TTA 对 self 与 TT100K **都有正收益**（多尺度天然提召回）：self det@0.5
   +4.9pp、TT100K E2E +2.7pp。TT100K 相对 smoke1 的 −6pp 来自颜色增广（非 TTA），
   TTA 反而把它收窄了一点。
3. **代价**：FP 上升（self 12→33、TT100K 145→226），且推理 ~3×
   （RDK X5 ≥10Hz 预算须重新评估）。self 漏到速度类的 FP 仍低（0.010/img）。
4. 剩余失败：`_4` pl30（事件 4，连续 2 帧 + 1 帧错值 pl20）、`_9` pl40（事件 7，
   生命周期仅 1 帧 → 不可验）。

## 判断

- **TTA 是有效的部署侧缓解**（零训练，直接对因“尺度混叠”），可作为候选推理
  方案或离线复核手段；是否上线取决于板端算力与 FP 政策。
- 若想“不花 3× 推理”就获得同等的尺度鲁棒性，回到**训练期多尺度**
  （`multi_scale>0`，§8.5 TODO 2，需一次全量训练）；两者可二选一或叠加。

明细：`reports/detector_eval_size_self_cropcolor1_tta.md`、
`reports/e2e_self_cropcolor1_tta.md`、`reports/event_eval_self_cropcolor1_tta.md`、
`reports/detector_eval_size_tt_cropcolor1_tta.md`、`reports/e2e_tt_cropcolor1_tta.md`。
