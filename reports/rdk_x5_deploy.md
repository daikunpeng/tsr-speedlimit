# RDK X5 量化部署（PTQ，hb_mapper 权威路径）

- 生成：2026-09-28
- 模型：detector `tt_cropcolor1`（YOLOv5nu，anchor-free）+ classifier `v3`（TinyCNN 245k，28 类）
- 工具链：OpenExplorer 1.2.8（docker，`hb_mapper` 1.24.3 / hbdk 3.49.15 / horizon_nn 1.1.0）
- 目标板：RDK X5，RDK OS 3.5.0，运行时 `hrt_model_exec` 1.24.5（HBRT 3.15.55.0）

## 结论摘要

| 项 | 结果 |
|----|------|
| 检测器 .bin | `det_cropcolor1_352x640_bayese_640x352_nv12.bin`（3.0 MB，NV12 352×640 输入） |
| 分类器 .bin | `cls_v3_64x64_nv12_64x64.bin`（461 KB，NV12 64×64 输入） |
| 检测器 FPS | **172**（单线程单核，纯 BPU forward） |
| 分类器 FPS | **1898**（0.52 ms） |
| 端到端 FPS | **85.5**（det+crop+cls，Python `hbm_runtime` 编排） |
| self det recall@0.5 | **37.3%（69/185）**（@conf0.25；见下方置信度偏移） |
| self e2e（det+cls） | **33.5%**（@conf0.25，cls 89.9%） |
| self FP/img | 0.001–0.002（@conf0.25） |

**量化本质无损**：板端 @conf0.25 的召回/误报与主机 fp32 @conf0.5 完全一致
（self det recall@0.5 37.3%、E2E 34.1%），唯一差异是**检测置信度整体下移**（见 §4）。

## 1. 工具链与编译

pip 版 `rdkx5-yolo-mapper` 依赖链拉取 torch 670MB 且走 PyPI 镜像仅 ~0.5 MB/s，改用
**OpenExplorer docker**（阿里云 OSS 6.4 MB/s，1.8 GB 约 5 分钟）：

```bash
wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz
docker load -i docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz
# image: openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8-py310
```

### 检测器（ultralytics YOLOv5u 官方 recipe）

- 导出：`scripts/rdk/export_monkey_patch.py --pt tt_cropcolor1/best.pt --imgsz 352 640`
  （monkey-patch Detect 头 → 输出 `[cls, box] × 3` 原始 head，DFL/解码在板端 host 侧）
- 校准 + 编译：`scripts/rdk/mapper.py`（march `bayes-e`，NV12 输入，data_scale 1/255，
  calibration `default`，20–50 张 TT100K train 图）
- 产物：6 个 float 输出（3 尺度 × {cls(1ch), bbox(64ch)}）

### 分类器（自定义 TinyCNN，通用 hb_mapper）

- 导出：`scripts/rdk/export_cls_onnx.py`（静态 batch=1，(1,3,64,64)）
- 编译：`scripts/rdk/mapper_cls.py`（march `bayes-e`，NV12 输入，
  `data_mean_and_scale` mean 127.5 scale 1/127.5，即 `(x/255−0.5)/0.5`）
- 校准：50 个 64×64 crop（38 speed + 12 neg）
- 踩坑：首版 `input_type_rt: rgb` 编译成功但板端 `hbm_runtime` 报 dtype 不匹配
  （RGB 输入要 int8），改用 **NV12**（与官方一致、uint8 天然）即通。

## 2. 板端冒烟

- 输入 640×352 自采图（trip `_5`，GT pl20，框 [433,112,471,147]）：
  板端检出 `[434,112,470,147]`（IoU≈0.99）、`cls=pl20`，端到端正确。
- `hrt_model_exec perf`：det 171.96 FPS（5.79 ms）、cls 1892.8 FPS（0.52 ms）。

## 3. self 域评估（982 帧 / 185 GT，`scripts/rdk/eval_board.py`）

| 阈值 | det recall@0.5 | FP/img | e2e（det+cls） |
|------|---------------:|-------:|---------------:|
| conf 0.5 | 22.2%（41/185） | 0.000 | 90.2%（37/41） |
| **conf 0.25** | **37.3%（69/185）** | 0.002 | **89.9%（62/69）→ 33.5% overall** |

对照主机 fp32（cropcolor1+v3，`reports/detector_cropcolor_ablation.md`）：
self det@0.5=37.3%（69/185）、E2E=34.1%。

## 4. 重要发现：检测置信度整体下移（int8 头部量化）

- 板端 @conf0.25 的召回（37.3%）== 主机 fp32 @conf0.5 的召回（37.3%），逐框一致。
- 根因：hb_mapper 把检测头 `cv3`（cls logits）也 int8 量化（ORT 模拟时 `nodes_to_exclude`
  head FP32 故无此偏移）。TT100K 校准的 logit 尺度对 self 域低 logit 检测精度不足，
  使 sigmoid 置信度系统性下移约一个「0.5→0.25」档。
- 尝试 `set_model_output_int16`：cls 输出 cos-sim 反而略降（0.996→0.990），不采用。
- 加校准图（30→200→50 抽样）不改变偏移：**偏移是头部 int8 量化固有**，非校准不足。
- **部署对策**：板端 conf 阈值用 **0.25**（等价主机 0.5）；FP 仍极低（0.002/img）。
  如需与主机完全一致的分数，后续可考虑 self 域无标签图做激活校准（须用户确认，涉
  测试域数据）或时序跟踪多帧投票（§8 TODO 1）。

## 4b. cls-head int16 尝试（2026-09-28 补充）

hb_mapper 节点精度只有 `int8`/`int16`（无 fp32，BPU 卷积硬件不支持 fp32）。用
`model_parameters.node_info` 把 cls 头最后 3 个 conv（`/model.24/cv3.{0,1,2}/cv3.{i}.2/Conv`）
设为 int16，重编译（`mapper.py --cls-head-int16`）：

| 版本 | det@0.5 | det@0.25 | det FPS |
|------|--------:|---------:|--------:|
| int8 head（默认） | 22.2% | 37.3% | 172 |
| cls-head int16 | 23.8% | **39.5%（73/185）** | 163 |

结论：int16 head 小有增益（@0.25 +2.2pp、@0.5 +1.6pp，代价 FPS −5%），
**但不消除置信度偏移**（0.5 阈值召回仍远低于 37.3%）。偏移主因是整条 head 特征/
box 分支 int8 + 域低 logit，非单点 logit 精度。产物存档 `runs/rdk/out_det_int16head/`，
不替代默认；是否采用（+2pp 换 −5% FPS）由业务定。

## 5. 部署产物与脚本

- `.bin`：`runs/rdk/out_det/`、`runs/rdk/out_cls/`（已 scp 至板 `/home/sunrise/tsr_models/`）
- 推理：`scripts/rdk/deploy_tsr.py`（自包含 cv2/numpy/hbm_runtime，det→crop→cls）
- **直采**：`scripts/rdk/deploy_tsr_cam.py`（`hobot_vio` libsrcampy 直取 NV12，见 §5b）
- 评估：`scripts/rdk/eval_board.py`
- 转换：`scripts/rdk/export_monkey_patch.py`、`mapper.py`、`export_cls_onnx.py`、`mapper_cls.py`
- 关键非方形适配：官方 `decode_boxes` 假定方形输入（`gen_anchor(grid_size)`），
  `deploy_tsr.py`/`eval_board.py` 已改为非方形 grid（`gh, gw` 分别 44×80 / 22×40 / 11×20）。

## 5b. hobot_vio 直采 NV12（2026-09-28）

板端相机（MIPI csi0）实测为 **sc132gs-hdr-2lane**，native **1088×1280（竖屏）raw10 30fps**
（DOL2 HDR；注意当前连的并非目标 GS130WI，若 GS130WI 是另一模组需重新确认分辨率/驱动）。

- 用 `libsrcampy.Camera`：`open_cam(0,-1,fps,640,352)` 后 `get_img(2,640,352)` 直接返回
  ISP 硬件下采样后的 **640×352 NV12**（dns0 通道 `roi 1088×1280 → target 640×352`）。
- `deploy_tsr_cam.py` 把该 NV12 直接喂 detector（**免 BGR→NV12 转换与软件 resize**），
  实测 **29.8 FPS**（被相机 30fps 上限封顶，非算力瓶颈）。
- 注意：1088×1280 → 640×352 是**非等比下采样**（0.85→1.82 形变），且无去畸变；
  真正部署须接入 ISP LDC/去畸变与正确 crop（同 §8 TODO「无去畸变前原图、ISP 参数不可调」）。
  当前脚本是直采通道打通验证，非最终成像管线。

## 6. 后续（TODO）

1. TT100K 干净域板端回归（需 640×640 方形编译或接受 352×640 形变口径）。
2. 分类器 RGB 输入 int8 语义 / NV12 色度下采样对小牌分类的影响量化。
3. 若部署需与主机分数对齐：self 无标签激活校准（须确认）或时序跟踪投票。
4. ~~hobot_vio 直采~~ 已打通（§5b）；剩余是去畸变/正确 crop + GS130WI 模组确认。
5. cls-head int16 是否采纳入默认（+2pp @0.25 换 −5% FPS）。
