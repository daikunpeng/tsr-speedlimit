# 项目策略（来自目标与设计讨论）

权威来源：
- [`target.md`](target.md) — 部署目标与性能约束
- [`TSR-chat-history.txt`](TSR-chat-history.txt) — 训练/数据设计讨论全文
- 本文件 — 落地成可执行原则（Agent 与后续实现以此为准）

---

## 1. 最终目标（部署约束优先）

| 项 | 要求 |
|----|------|
| 中级目标 | **YOLOv5 级检测器** + **极小分类器**，中国场景限速标志检测与识别 |
| 板端 | 地瓜派 **RDK X5** |
| 相机 | GS130WI，约 120° 宽角，畸变大；推理图为**已去畸变** |
| 输入分辨率 | **640×352** |
| 在线帧率 | ≥ **10 Hz** |
| 离线目标 | 尽量 ≥ **60 Hz**（留出余量） |

原则：**官方生态成熟度 > 排行榜多几个点**。优先 YOLOv5n/s 及板端已验证的转换/量化路径；不默认追 YOLO8/11/Transformer，除非板端已验证可用。

训练可以在 RTX 3060 上用更大 teacher；部署只保留小模型。

---

## 2. 系统架构（两级，不要单模型细分类）

```
Camera Image (640×352, undistorted)
        │
        ▼
┌───────────────────┐
│ Detector (YOLOv5n/s) │  类别尽量简单：speed_limit_sign（可选 other_sign）
└─────────┬─────────┘
          │ bbox
          ▼
     crop ROI
          │
          ▼
┌───────────────────┐
│ Tiny Classifier     │  20/30/40/50/60/70/80/...
└───────────────────┘
```

原因：
- 检测只要认出「限速牌」；分类要读数字——特征需求不同。
- 远距离：能检出但易分错；近距离才易分对 → 需 object-centric + 时序，而不是把所有类别塞进一个 detect head。

板端推理还可加：
- **Tracking** → sign lifecycle
- 选最清晰几帧 → classifier → **多帧投票**
- 多数帧无牌时 classifier 几乎不跑，算力可控。

---

## 3. 核心问题：防 shortcut / 跨相机泛化

公开数据（TT100K 等）与自采（GS130WI）混训时，模型容易学：
背景、相机风格、FOV、清晰度、道路类型 → 类别，而不是「红圈 + 数字」。

策略主线：

**多域数据 → 削弱域偏差 → 强化牌面局部特征 → 模拟真实退化 → 分阶段训练 → 按域验证 → Hard Case 闭环**

优先手段（工程收益高）：
1. Detector / Classifier **解耦**
2. 全部做成 **sign-centric crop**（tight / 1.3–1.5× / 2.0× 等）
3. **Small-object degradation**（缩到 8/12/16/24/32 px 再放大）
4. Copy-Paste / background swap，打断「背景↔类别」相关
5. 按 Camera / Road / Trip / Physical Sign 做 **OOD split**，禁止视频连续帧随机划分

核心 KPI（比单一 mAP 更重要）：
- **sign 像素尺寸 → 分类准确率曲线**
- 按 domain 的 Recall / Classification 矩阵
- Background-swap 时 P(class) 是否稳定

---

## 4. 数据阶段顺序（先数据，后定模型细节）

当前阶段：**只在 TT100K 上工作**；`data/` 预留目标场景测试集（尚未就绪）。

推荐顺序：

1. 收集 / 下载（现：TT100K）
2. 统一标签体系（保留原始 label 可追溯）
3. 每条样本带 **domain metadata**（source、camera、resolution、trip…）
4. 生成 sign crop
5. **数据体检**（类别数、尺寸分布、长尾、重复帧…）
6. 再定训练超参与增强强度

第一版 unified label（示例）：

- 检测：`speed_limit_sign`（pl* 最高限速 + il* 最低限速；pm* 限重已移出，见 AGENTS.md 语义修正）
- 分类：`speed_limit_20` / `30` / `40` / …（由 TT100K 的 `pl*` / `il*` 映射；pl/il 分组）

TT100K 过滤范围见 `AGENTS.md`：正类 `pl*`（最高限速）+ `il*`（最低限速）；`pm*`（限重）移入 `data/TT100K_weight_pm/` 留档。

增强（数据体检后再加强）：small-object degradation、blur、JPEG/ISP、畸变随机化、domain-balanced sampling 等。**现阶段不要一上来堆大量 augmentation。**

---

## 5. 目录角色

| 路径 | 角色 |
|------|------|
| `data/TT100K/` | 公开 TT100K 全量；后续 crops 可挂旁边 |
| `data/TT100K_speed_det/` / `TT100K_speed_mc/` | 限速过滤后的检测单类 / 多类子集（pl*+il*） |
| `data/TT100K_weight_pm/` | **限重归档（pm*，非速度）**：hard-negative 备用 |
| `data/self/` | **目标场景自采测试数据**（GS130WI 等）；现空，就绪后放入 |
| `docs/` | 目标、策略、环境、聊天记录 |
| `configs/` / `scripts/` / `runs/` / `weights/` | 配置、脚本、实验、权重 |

---

## 6. 分阶段路线图（摘要）

| 阶段 | 内容 | 现状 |
|------|------|------|
| A | 仓库规则、环境文档、策略对齐 | 进行中 |
| B | TT100K 下载 → 限速过滤 → 标签统一 → crop → 数据体检 | 下一步 |
| C | Detector（单类/少类）+ Tiny classifier 在 TT100K 上验证 | 待定 |
| D | 自采 `data/` 接入；domain 指标；Copy-Paste / OOD | 等自采就绪 |
| E | 量化 / 转板端；RDK X5 + 640×352；跟踪与生命周期投票 | 部署期 |

---

## 7. Agent 执行注意

- 改架构或部署目标：先改 `target.md` / 本文件 / `AGENTS.md`，再改代码。
- 全量下载 TT100K、全量训练、长时间实验：先征得用户确认。
- 默认模型选型偏向 **YOLOv5n/s + tiny CNN**，不要擅自换成 YOLO11「因为更新」。
