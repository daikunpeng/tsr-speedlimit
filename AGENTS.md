# AGENTS.md

Traffic Sign Recognition（TSR）：中国场景 **限速标志** 检测与识别，最终部署到 **地瓜派 RDK X5**。

必读：[`docs/target.md`](docs/target.md)、[`docs/strategy.md`](docs/strategy.md)、[`docs/local_environment.md`](docs/local_environment.md)。
设计讨论原文：[`docs/TSR-chat-history.txt`](docs/TSR-chat-history.txt)（参考用；冲突时以 `target.md` / `strategy.md` / 本文件为准）。

## 背景：Cursor → OpenCode 交接

- 本项目此前由 Cursor 创建并推进；现转由 OpenCode 接手。
- `.cursor/rules/*.mdc` 保留，其中有价值约定已并入本文件；**冲突处以本文件为准**。
- 已知冲突：`.cursor/rules/python-ultralytics.mdc` 写的入口 `uv run python` **已废弃**——本机 `uv run` 会重解析依赖、可能装错（CPU 版）torch。一律 `.venv/bin/python scripts/...`。

## 目标与约束

- 中级：`YOLOv5n/s` 级 **detector** + **tiny classifier**（两级流水线）；冒烟权重 `weights/yolov5nu.pt`
- 板端：RDK X5；输入 **640×352**（GS130WI，推理图已去畸变）；在线 ≥10 Hz，离线目标 ≥60 Hz
- **部署约束优先于追求更新更大的检测器**；不默认改用 YOLO8/11/Transformer

## 架构（必须遵守）

```
图 → Detector（优先只检 speed_limit_sign）→ crop → Tiny Classifier（限速数值）
```

可选后期：tracking → lifecycle → 多帧投票。
不要把「是否限速牌」和「40/60/80…」全部塞进同一个 detect 多类 head 作为默认方案。

## 当前阶段：数据集整理

1. TT100K 已下载到 `data/TT100K`（YOLO 格式）；限速过滤已完成（产物见下）
2. **标签审计已完成（2026-09-23）**：修正 pm/il 语义颠倒（pm=限重、il=最低限速）并重建过滤子集；颜色审计误报经用户全量复核证伪（蓝底=高速背板/夜景偏色），**确认 TT100K 标注无误、未剔除任何样本**，复盘见 `reports/label_audit.md`
3. sign crop 已按新语义重建：`data/TT100K_speed_crops/`（pl*+il*，7734 sign ×4 scale = 30936 crops + `manifest.csv`），体检报告 `reports/crop_stats.md`；tiny/暗图样本全部保留（贴近部署场景）
4. **smoke 训练已完成（2026-09-23）**：
   - detector `yolov5nu` 100ep@640 → `runs/detector/smoke1/`，val best mAP50≈0.73；
     test 尺寸分桶 **recall@0.25=92.4%**，目标区间 12–28px **97.6%**，FP/img@0.25=0.11
     （`reports/detector_eval_size.md`，原始明细 `runs/eval/smoke1_test/`）
   - tiny classifier 245k 参数 → **test acc 94.9%**，各尺寸桶 ≥88%
     （`reports/classifier_smoke.md`，`runs/classifier/smoke1/`）
   - 主混淆：`il80→il60`（22 次）、相邻限速值互混；增强/难例挖掘待做
5. **E2E 级联评估已完成（2026-09-23）**：`reports/e2e_eval.md`（conf≥0.5，20% pad）
   - 总体 e2e **82.7%**（det 86.4% × cls **95.6%**）；目标区间 12–28px **91.7%**（det 96.2% × cls 95.3%）
   - det crop 下 cls acc 95.6% ≈ GT crop 基线 94.9%——det 框质量接近 GT（IoU P50=0.90），
     **不存在训练/部署框失配**；剩余混淆为相邻限速值互混（il80→il60×16 等）
   - Det FP 155 个（0.051/img，pm 限重/限高等）被 classifier 强制分到速度类（`pl30/50/40` 居多）
   - **踩坑记录**：首版 e2e 曾因 cv2.imread(BGR) 直接喂 PIL(RGB) 训练的 classifier，
     il 蓝底牌 88% 错分为 pl（e2e 假低至 69.9%）；`cvtColor(BGR2RGB)` 修复后 il→pl 跨组错误 265→2。
     教训：**cv2=BGR、PIL=RGB，跨库混用必须显式转换**
6. **A 方案已完成（2026-09-23，C 方案落地）**：classifier v2（28 类含 `other`）
   - 负样本 `data/cls_neg/`（6672 crop：pm/pr/ph 标签 crop ×4 scale + det FP crop），
     采集脚本 `scripts/collect_neg_crops.py`；正样本用满 4 scale（30936 crop）
   - `reports/classifier_v2.md`：speed acc **96.4%**（s1.5 子集 97.0%）、label 负样本拒绝 98.7%
   - `reports/e2e_v2.md`：e2e **84.2%**、目标区间 **93.7%**；det FP 拒绝名义 61%，
     经用户目检 155 个"FP"中 **~46 为 TT100K GT 漏标真限速牌** →
     **真拒绝率 ~86%**、真有效 FP **~0.005/img**，三项验收全部达标
   - v2b（detfp×8 加权）无增益已弃用；**v2 为当前最优**（`runs/classifier/v2/best.pt`）
   - 附带发现：train/val detfp 负样本估计 ~30% 为 GT 漏标真牌（污染可控，
     speed→other 误杀 0.58%）；v3 负样本清洗为候选优化，不阻塞
7. **下一步：B（X5 导出冒烟）→ C（自采数据）**
   - B：detector ONNX @640×352 + classifier ONNX @64×64 + parity 校验；板子可用
   - C：`data/self/`（GS130WI 640×352）部署域真实验证，最终验收关
   - 候选优化（按需）：v3 负样本清洗、il 相邻值难例挖掘、多帧投票
8. `data/self/` 预留自采测试集（GS130WI），现为空；就绪前不做依赖自采数据的训练主路径

### 限速过滤产物（2026-09-23 语义修正后重建）

| 路径 | 用途 |
|------|------|
| `data/TT100K_speed_det/` + `configs/TT100K_speed_det.yaml` | 检测：单类 `speed_limit_sign`（仅 pl*+il* 框） |
| `data/TT100K_speed_mc/` + `configs/TT100K_speed.yaml` | 多类限速：**pl*+il*，29 类**（plx/ilx 无实例），id 从 0 连续 |
| `data/TT100K_weight_pm/` + `configs/TT100K_weight_pm.yaml` | **限重归档（pm*，非速度）**：hard-negative 备用，勿当正类训练 |
| `data/TT100K_speed_meta/class_map.json` | speed 类映射（old↔new、semantic、speed_value） |
| `data/TT100K_speed_meta/weight_class_map.json` | weight 类映射（speed_value 为吨） |
| `data/TT100K_speed_meta/label_suspects.json` | ~~已废弃~~ → `color_flags.json`（信息性颜色标记，非错误清单） |
| `reports/speed_filter_stats.md` | 过滤统计 |
| `reports/class_distribution.md` | pl/il 与各限速值分布报告（图在 `reports/figures/`） |
| `reports/label_audit.md` | 标签审计报告（语义修正证据 + 可疑清单） |
| `scripts/filter_speed_limit.py` | 可重复执行（`--dry-run` / `--force` / `--exclude <json>`） |

注：过滤子集 `images/<split>/` 为**真实目录 + 逐文件软链**指向 `data/TT100K/images`（不占磁盘）。
**禁止**把 `images` 做成整目录软链：ultralytics `check_det_dataset()` 对路径 `.resolve()`
会跟到原始 `TT100K/labels`（221 类），单类集会把全部正样本标成 corrupt 丢掉
（2026-09-23 已踩坑，`filter_speed_limit.py::ensure_images_tree` 已修正）。

### 数据约定（自 .cursor/rules 并入，仍有效）

- 每条样本尽量保留 domain 元数据：`dataset_source` / `camera_id` / `resolution`（可空）
- 先 label 统一 + crop + 数据体检，再上大规模 augmentation
- 过滤后 class id 从 0 连续编号，与 YAML `names` 一致；保留原始 TT100K 类名映射便于追溯
- 数据集 YAML 放 `configs/`（`path` 为绝对路径）；勿改 Ultralytics 包内原始 yaml
- **以数据自带标签为准**：视觉统计规则只能做体检，不能做剔除判定；任何数据改动必须先经用户人眼确认
  （2026-09-23 教训：颜色审计曾把「蓝底背板/夜景偏色」误判为错标，经用户全量复核证伪，复盘见 `reports/label_audit.md`）
- **尺寸/坐标系统一按 @640 等效像素报告**（TT100K 2048 方图 ×0.3125；部署 640×352 用保持宽高比的 scale）
  — 检测/分类的尺寸分桶、target 区间 12–28px 均是 @640 口径；源图**禁止离线 resize**（归一化标签会与 ultralytics 在线缩放失配）

## 限速类别（TT100K 前缀，语义已修正 2026-09-23）

| 组别 | 前缀/名 | 含义 | 外观 |
|------|---------|------|------|
| 最高限速 | `pl*`（含 `plx`） | 禁止超过的限速 | 白底红圈黑数字 |
| 最低限速 | `il*`（含 `ilx`） | 最低速度 | **蓝底**白数字，值 50–110 |
| 限重（非速度） | `pm*`（含 `pmx`） | 限制质量，值为**吨**（1.5–55） | 白底红圈黑数字+t |

**重要**：早期文档把 pm 当「最低限速」、il 当「限速信息」——语义颠倒（2026-09-23 经用户目检
`1113.jpg pm20`=限重 20t + 全量颜色统计证实，见 `reports/label_audit.md`）。pm* 不属于限速类，
已移入 `data/TT100K_weight_pm/` 留档（hard-negative 备用，勿当正类训练）。
`plx` / `pmx` / `ilx` 在 TT100K 中无实例。

不要默认扩到全量 221 类。扩类先改本文件与 `docs/strategy.md`。
检测阶段标签可先合并为 `speed_limit_sign`；分类阶段再映射到具体速度值。

## 目录与 git

```
docs/        # target / strategy / setup / env / 聊天记录
data/        # TT100K 全量 + TT100K_speed_* 过滤子集 + self/ 自采预留
configs/     # 数据集 YAML
scripts/     # check_env / download_tt100k / filter_speed_limit
runs/ weights/ reports/   # 实验产物、权重、报告
```

`.gitignore` 已排除：`uv.lock`、`data/**`（除 README）、`runs/`、`weights/`、`*.pt`、`.wheels/`。勿提交大数据/权重。

## 环境（RTX 3060 12GB，CUDA toolkit 12.4）

- 安装与下载步骤：[`docs/setup.md`](docs/setup.md)
- Python：**uv** + 3.11；统一 `.venv/bin/python scripts/...`（不要 `uv run`；不要用系统 Python）
- 环境冒烟：`.venv/bin/python scripts/check_env.py`（CUDA + yolov5nu）
- torch 2.6.0+cu124 轮子来自阿里云镜像（`.wheels/`，已 gitignore）；不要升级 NVIDIA 驱动（桌面会话依赖）
- 代理（易踩坑）：国内镜像（阿里云/清华/TT100K）先 `unset` 全部 proxy 直连；仅 GitHub 等海外源用 `http_proxy=http://127.0.0.1:7890`；勿依赖 `ALL_PROXY=socks://`（坑 uv/httpx）
- 可用显存约 11GB（桌面占用约 0.7GB）；OOM 先降 `batch`

### GPU 任务与 shell 安全（2026-09-23 系统卡死强制重启的教训）

- GPU 任务启动前先 `nvidia-smi` 确认无残留 CUDA 进程；**同一时刻只跑一个 GPU 任务**，勿并行叠加重试
- 长任务：`nohup .venv/bin/python -u ... > log 2>&1 & echo $! > runs/<task>.pid`，用短 sleep 循环轮询日志，**勿长时间 sleep 挂住工具**
- **禁止 `pkill -f <脚本名>`**：`bash -c` 命令行里含同样字符串，pkill 会匹配到执行中的 shell 自身（表现为命令挂起直到超时）。终止进程一律 `kill $(cat runs/<task>.pid)`
- ultralytics `predict` 的 `source` 传**目录字符串**（`stream=True` + wanted-set 过滤），**勿传超大路径 list**（3067 路径曾致进程挂死一小时零进度）
- 大量小输入的推理必须 **batch 化**（收集后 256/batch forward），勿逐样本调 GPU（上千次小 kernel 会与桌面会话抢卡）

## 训练与验证原则（摘要）

- Object-centric crop；核心 KPI：`sign 像素尺寸 → 分类准确率`
- 按 domain / camera / trip 验证；禁止视频近邻帧随机 split
- 多域时 domain-balanced sampling；用 background swap 检查是否依赖背景
- 全量下载 / 全量训练 / 长时间实验：先征得用户确认
- 尚无测试/lint 配置；验证以脚本冒烟 + 数据体检报告为主
- 详情见 [`docs/strategy.md`](docs/strategy.md)

## Agent 行为

- 回复中文、简洁
- 不主动 commit / push
- 改目标/架构先改文档再改代码
- 阻塞写 `reports/BLOCKED.md`，停止无效重试
