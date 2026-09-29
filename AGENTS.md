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
7. **B0 + B1-light 已完成（2026-09-23）；C 准备件已交付**
   - B0 `reports/export_parity.md`：ONNX 导出 opset12（`runs/export/det_sq640.onnx`、
     `det_deploy_352x640.onnx` 输入 (1,3,352,640)、`cls_64.onnx` 动态 batch）
     —— **FP32 与 torch 完全等价**（cls argmax 99.99%，det/e2e 仅 conf=0.5 边界 1 框差，e2e 同 84.2%）
   - B1-light `reports/quant_sim.md`：ORT INT8 QDQ 模拟（校准 200 det 图 + 512 cls crop）
     —— **e2e 掉约 1pp**（84.2→83.3）、目标区间 −0.9pp、cls 无损；
     **踩坑：朴素全量 INT8 使 score 通道全 0 → recall 0%**，必须
     `nodes_to_exclude=/model.24/*`（head FP32）
   - C 准备件：`data/self/classes.txt`（pl/il/pm/pr/ph/pne 82 类 + `check_me`）+
     `scripts/prelabel_self.py`（det conf0.25 imgsz=(352,640) + cls v2 → YOLO txt，
     与 e2e 同前处理，冒烟对 GT pl40 全中）——用户用 X-AnyLabeling 导入该 classes 文件
     加载预标注修正；纯测试集，无 train/val split
   - **C 数据落位 + 评估参数化已完成（2026-09-23）**：
     - 首批 61 帧 + 2026-09-23 扩标 5 trip（_3/_4/_5/_7/_9）→ 现 **982 帧 / 216 JSON /
       185 det 框 / 95 lookalike（含新类 `pne`=禁止驶入硬负样本 77 个，已入 classes.txt
       与 lookalike 路由）**；`prep_self_dataset.py --force` 重建，0 可疑；
       体检 `reports/self_dataset_stats.md`（78% det 框在 12–28px 目标带）
     - `eval_detector_size.py` / `eval_e2e.py` 已参数化（`--data-root/--src-size/--imgsz`
       等，e2e 按叶子目录 predict）；**TT100K 全量回归精确复现 92.4% / e2e 84.2%**
     - **self 域基线（982 帧扩标后，2026-09-23）**：
       det recall@0.5=**11.9%**（22/185）、@0.25=18.4%，FP@0.5 仅 0.007/img（问题是召回不是误检）；
        尺寸效应显著：24–32px @0.5=43%、32–48px=100%（4/4）、≥48px=33%（1/3），
        **12–16px=0%、16–24px=3%**，
       目标带 12–28px @0.25 仅 12.4%；e2e@0.5=**9.7%**（cls 81.8% n=22）；
       分数扫描 `reports/self_score_sweep.md`：GT conf 中位 0.00165、**68/185 连定位都丢**、
        分数中段 GT/FP 重叠 → 单靠降阈值难兼顾总体召回/误报
        （注：用户声明部分非限速牌可能漏标，少量"FP"或为真漏标）
       （`reports/detector_eval_size_self.md`、`reports/e2e_self.md`）——
       TT100K 冒烟权重对 GS130WI 域适配差，C 方案核心结论
   - ORT 环境：`onnxruntime-gpu==1.19.2`（1.23 需 libcublasLt.so.13 不存在）；
     跑前必须 `export LD_LIBRARY_PATH=$NV/cublas/lib:$NV/cudnn/lib:...`（nvidia 轮子目录，
     配方见 `reports/export_parity.md`）
   - **新优化方向（用户 2026-09-23 确认，详见 `docs/strategy.md` §8）**：
     业务验收改为路牌接近时逐事件稳定读对：短边 ≥24px 的连续 5 帧中
     ≥3 帧 E2E 正确、无同一错误值的稳定输出；同时报告可验事件覆盖率。
     自采数据少、分布不均且非速度标注不全，当前仅作开发评估；
     先试冻结权重的轻微输入预处理，再做 TT100K-only 的缩放后涂抹/局部拉伸
     重采样增广，逐项评估 GS130WI 与 TT100K；分类与时序放后。
     无去畸变前原图，ISP 参数不可调；全量训练须另获用户确认。
     事件评估 `scripts/eval_self_events.py` / `reports/event_eval_self_baseline.md`：
     自动关联 **10 个实体牌事件**（其中 9 个峰值≥24px；关联需人工复核），
     严格 ≥24px 连续 5 帧窗口仅 **3/10 可验、2/3 通过**；其余窗口不足，不算失败。
     轻量预处理试验 `reports/preproc_pilot.md`：锐化/亮度 CLAHE 冻结权重 E2E
      **9.7→7.6/8.1%**，事件均 2/3；不默认加入部署。
      用户批准的 **一次 TT100K-only 全量检测器训练对照已完成**（`tt_smear1`，100ep，
      `reports/detector_smear_ablation.md`）：局部压缩/重采样模拟涂抹、val 不增强；
      TT100K test E2E **84.2→81.0%**，self E2E **9.7→10.3%**，
      self 严格可验事件 **2/3→1/3**，不替换 smoke1。smoke1 冻结权重
      self conf0.25 则 3/3 通过，但只覆盖 3/10 个事件，13 个名义速度 FP 均 <12px；
      暂仅为开发集候选阈值/尺寸门控，非独立验收。**本次全量训练授权已使用**；
      后续全量训练另行确认。
   - **冻结权重诊断（2026-09-24，`reports/frozen_diag.md`）**：
     检测器主因是**牌面色彩/渲染偏移**（`_4` 过曝发白、`_9` 暗红几乎无定位；
     `_3` 标准外观 det conf 0.9）；分类器同样受影响——`_7` 暗红 pl40、
     `_5` pl20 的 GT crop 处于 `other` 决策边界（1px 行偏移即 other 0.72→pl40 0.95
     翻转），24–48px GT-crop 正确仅 25/34；位置有影响（上缘 conf P50 0.37 vs
     中部 0.70）但与颜色共变。
   - **位置/画幅裁切训练已完成（2026-09-24，用户批准的第二次全量对照）**：
     `tt_crop1`（`scripts/train_detector_crop352.py`，100ep；训练期 640×352
     在线裁切 + 50% 概率把 GT 牌置顶带，标注同步平移/裁剪/删除）。
     `reports/detector_crop_ablation.md`：self det recall@0.5 11.9→**16.2%**、
     24–32px @0.5 43→**60%**、无定位 68→**55**、GT conf P50 ×2.2——self 域
     最好检测器；但 cls v2 把新召回的暗红牌判 `other`（det TP 上 cls 56.7%），
     E2E 9.2%、事件 2/3 未超 smoke1（conf0.25 为 2/3 vs smoke1 3/3）。
     TT100K E2E 84.2→80.6%。**级联未达标，暂不替换默认权重**；
     tt_crop1 记为 self 域检测器候选，等 cls 颜色增广落地后合并决策。
     瓶颈已移到分类器；下一步候选（均须用户确认）：cls v3 牌面局部颜色/
     曝光增广（分钟级小训练）→ 视结果再议检测器侧颜色增广全量训练。
   - **cls v3 已完成（2026-09-24，用户批准的颜色/曝光增广小训练）**：
     v2 同数据/同架构/同参数，唯一新增训练期牌面颜色/曝光增强（概率 0.5，
     hue 不动，test 不增强；`runs/classifier/v3/best.pt`）。
     self GT-crop 分类 112→**159/185**（暗红 `_7` 8→19/20、`_5` pl20 15→23/24、
     `other` 误杀 51→14）；`reports/cls_v3_cascade.md`：
     **tt_crop1 + v3** self E2E 9.7→**15.1%**、24–32px E2E **53.3%**、
     事件 **3/3**（`_5` pl20 5/5 通过）；TT100K smoke1+v3 **84.4%**（不降）、
     crop1+v3 80.4%。建议 tt_crop1+v3 为目标域默认组合候选（待用户确认替换）。
     剩余：self 55/185 无定位（`_9`/`_4` 暗/过曝小牌）→ 检测器侧颜色增广
     为下一候选全量训练（另行授权）。
   - **检测器侧颜色增广已完成（2026-09-24，用户批准 A→B 计划的第三次全量对照）**：
     先 A（漏检帧诊断 `reports/diag_missed_frames.md`：大牌漏检主因是色彩——
     漏检组红度 P50 6 vs 检出 30），再 B（`tt_cropcolor1`，crop352 配方上叠加
     PlateColorAug 牌面局部颜色/曝光增广，概率 0.5、hue 不动、框几何不变；
     `scripts/train_detector_crop_color.py`）。`reports/detector_cropcolor_ablation.md`：
     **cropcolor1 + v3** self det@0.5 11.9→**37.3%**、无定位 68→**11**、
     GT conf P50 ×100、E2E 9.7→**34.1%**、24–32px @0.5 **83.3%**、
     事件 3/3、近距曾读对 8/9；代价 TT100K E2E 84.2→**75.5%**（−8.7pp，
     干净 <8px/≥48px 牌置信下降，12–28px 目标带仍 93.4% recall）。
     目标域最大单一杠杆；默认组合候选 **tt_cropcolor1 + cls v3**。
     可选后续（另行授权）：颜色增广降概率/收窄强度、或仅 12–48px 目标带增广
     以找回 TT100K 干净牌；仍需新 trip 独立验收。
   - **验收标准修正 + 尺度敏感性发现（2026-09-24）**：验收改为生命周期内
     **连续 3 帧检出且分类正确**（取消 ≥24px 窗口；<3 帧单列不可验），
     cropcolor1+v3 新标准 **7/10**。`_1` pl60 漏检判为**输入尺度敏感性（混叠）**
     而非外观：每帧都定位（IoU 0.89–0.97）但 15–20px 区间 conf 塌陷，换尺度即回升
     （`scripts/diag_scale_sweep.py` 多尺度推理）。**实验 A（颜色尺寸门控）暂停**，
     TODO 记入 `docs/strategy.md` §8.5：时序跟踪+多帧投票、多尺度训练/推理。
   - **TTA 零训练探针完成（2026-09-24，`reports/tta_probe.md`）**：`eval_detector_size.py`
     与 `eval_e2e.py` 加 `--tta`（predict augment=True）；冻结 cropcolor1+v3 下
     `_1` 变**连续 3 帧→pass**、事件 **8/9**、self det@0.5 42.2%、TT100K E2E 78.2%；
     FP 上升（self 0.034/img、TT100K 0.074/img）且推理 ~3×。TTA 记为部署侧候选。
   - **多尺度训练已试、被否（2026-09-24，已授权全量对照）**：`train_detector_crop_color.py`
     加 `--multi-scale`（Ultralytics 内置整图随机缩放）；`tt_cropcolor1_ms033`
     （multi_scale=0.33）`_1` 尺度塌陷**未消除**、self det@0.5 37.3→20.5%、事件
     7/9→3/9、TT100K E2E 75.5→74.1%（`reports/detector_multiscale_ablation.md`）。
     结论：整图 multi_scale 训练不能替代 TTA；`_1` 尺度塌陷的部署侧解法回到
     时序跟踪+多帧投票（首选）或 TTA。ms033 仅存档，不替换默认。
   - **RDK X5 量化部署已完成（2026-09-28，`reports/rdk_x5_deploy.md`）**：
     OpenExplorer 1.2.8 docker（hb_mapper 1.24.3，pip 版 torch 670MB 太慢弃用）→
     det `tt_cropcolor1`（monkey-patch head，NV12 352×640）+ cls `v3`（NV12 64×64）
     编译 `.bin` 上板。det 172 FPS / cls 1898 FPS / 端到端 85.5 FPS，全部达标。
     self det recall@0.5 **37.3%（69/185，@conf0.25）== 主机 fp32 @conf0.5 逐框一致**、
     e2e 33.5%、FP 0.002/img；**量化本质无损，仅检测置信度整体下移一个档位**（head
     cv3 也 int8 量化，`set_model_output_int16`/加校准图均不改善），**板端 conf 阈值
     用 0.25**。脚本 `scripts/rdk/`（export/mapper/deploy_tsr/eval_board，已修官方
     非方形 grid 假定）。补充：**cls-head int16**（`node_info` 把 3 个 cls 头 conv 设
     int16，BPU 无 fp32 可选）@0.25 +2.2pp（37.3→39.5%）换 −5% FPS，不消除偏移、存档
     不替代默认；**hobot_vio 直采已打通**（`deploy_tsr_cam.py`，`get_img(2,640,352)`
     直接返回 ISP 下采样 640×352 NV12，实测 29.8 FPS 被相机 30fps 封顶；当前连的是
     sc132gs-hdr 1088×1280 竖屏，非 GS130WI，且 1088×1280→640×352 非等比形变、无去畸变）。
      原 B2 板端冒烟已并入本次。TODO：TT100K 干净域板端回归、self 无标签激活校准（须确认）、
      去畸变/crop 成像管线 + GS130WI 模组确认。
   - **TSR ROS 节点已完成（2026-09-28，`reports/ros_node_design.md`）**：C++ 单节点
     `tsr_detect`（订阅自定义 `Nv12Image` 640×352 → det→crop→cls→自写 IoU/中心距离
     tracker→多帧投票→发布 `SpeedLimitSignArray`）+ 测试重放 `tsr_replay`（10Hz）。
     代码 `ros/tsr_node/`（板端 `/home/sunrise/tsr_ws/`，colcon 编译，libdnn/OpenCV/rclcpp）。
     Level 1 无 ROS 单测 `tsr_test` 与 Python `eval_board.py` **逐框一致**（det@0.25 37.3%、
     e2e 89.9%）；tracker N/M/K=3/3/3，`_5` 生命周期 new→confirmed→speed **pl20**→lost 正确。
     踩坑/约定：`hbm_img_msgs` 无通用可变尺寸消息（仅 480/540/1080P），故 640×352 用自定义
     `Nv12Image`（真实驱动需发布此消息或加适配）；可执行名 `tsr_detect` 不与接口包 `tsr_node`
     同名（rosidl target 冲突）；关联用中心距离门控（小牌快速移动时 IoU 会塌到 <0.3）。
     交给同事集成前：输出消息 schema 为占位；调参已外置 `config/tsr_node.yaml`（ROS2 参数
     score_thres/N/M/K/assoc_gate），det/cls 路径仍 CLI；crop 已优化为**先按框切 NV12 再转 BGR**
     （`nv12_crop_to_bgr`，COLOR_YUV2BGR_NV12，避免整帧转换）；`tsr_test --vis <dir>` 可离线画框存图。
 8. `data/self/` 现有 6 trip / 982 帧（GS130WI，见 7.），仍是**纯测试/开发评估集**；
   任何训练主路径不得依赖它（域适应/微调需先改 `docs/strategy.md` 与本文件）。

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
