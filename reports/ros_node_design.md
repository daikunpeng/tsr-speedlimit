# TSR ROS 节点设计与测试（RDK X5，TROS Humble）

- 生成：2026-09-28
- 交付：C++ 单节点 `tsr_detect`（订阅 NV12 → 两级推理 → 自写 tracker → 发布）+ 测试重放节点 `tsr_replay`
- 代码：仓库 `ros/tsr_node/`（板端 `/home/sunrise/tsr_ws/src/tsr_node/`）
- 依赖：`rclcpp`、`std_msgs`、OpenCV、`libdnn`（hbm_runtime）

## 0. 结论

| 项 | 结果 |
|----|------|
| Level 1（无 ROS 单测） | det recall@0.25=**37.3%（69/185）**、FP 0.002/img、e2e **89.9%** —— 与 Python `eval_board.py` **逐框一致** |
| Tracker 生命周期 | `_5` trip：new(59)→confirmed(61)→speed **pl20**(64)→lost(72)，GT 确为 pl20 ✓ |
| Level 2（ROS 集成） | `tsr_replay`(10Hz) → `tsr_detect` → 发布 `speed_limit_signs`，track_id=0 speed=**pl20**，状态机 0→1→2 完整 ✓ |

## 1. 业务流程图（节点内部）

```
订阅 Nv12Image (640x352 NV12)
   │
   ▼
① Detector:  libdnn(hb_dnn) BPU forward
   │  6 输出(3×{cls,box}) → DFL decode(非方形 grid 44×80/22×40/11×20) → NMS
   ▼  boxes(xyxy, conf)
② Crop:     每个框 NV12→BGR→crop→resize 64×64→BGR→NV12（仅在有检出时转 BGR）
   ▼
③ Classifier: BPU forward → 28 类 logits → argmax（27=other 拒绝）
   ▼  (bbox, det_conf, cls_id)
④ Tracker:  自写 IoU/中心距离关联 → 生命周期状态机
   ▼
⑤ Voting:   每 track 多数票 + 连续 K 帧同值 → 发 confirmed 限速值
   ▼
发布 SpeedLimitSignArray（全 track 状态，含 speed_value）
```

## 2. 消息定义（`msg/`）

- **输入 `Nv12Image`**：`header`、`uint32 width/height`、`uint8[] data`（packed NV12）。
  > 注：D-Robotics `hbm_img_msgs` 只有固定尺寸 480P/540P/1080P 消息、无通用可变尺寸图，
  > 640×352 非标，故用自定义 `Nv12Image`。同事接入真实驱动时，驱动应发布此消息
  > （或加一层适配）。后续如需零拷贝可再切 `hbm_img_msgs`/hbmem。
- **输出 `SpeedLimitSign`**：`header`、`uint32 track_id`、`float32 x1/y1/x2/y2`、
  `float32 det_conf/cls_conf`、`uint8 state`（0=candidate 1=confirmed 2=lost）、
  `string speed_value`（"pl20"/"il80"/"other"/""）。
- **输出 `SpeedLimitSignArray`**：`header` + `SpeedLimitSign[] signs`（每帧发布全 track 快照）。

## 3. 文件结构（`ros/tsr_node/`）

```
include/tsr/{detector,classifier,tracker,nv12}.hpp
src/{detector,classifier,tracker,tsr_node,tsr_replay}.cpp
test/test_tsr.cpp        # Level 1 无 ROS 单测（det+cls+track，落盘/打事件）
msg/{Nv12Image,SpeedLimitSign,SpeedLimitSignArray}.msg
CMakeLists.txt package.xml
```

- `tsr_core`（静态库）：detector / classifier / tracker / nv12 —— 推理与跟踪核心，ROS 无关。
- `tsr_detect`：ROS 节点（注意：可执行名不叫 `tsr_node`，避免与 rosidl 接口 target 同名冲突）。
- `tsr_replay`：测试重放节点。
- `tsr_test`：Level 1 单测可执行。

## 4. Tracker 状态机与参数（`tsr/tracker.hpp`）

```
CANDIDATE --连续N帧检出--> CONFIRMED --连续K帧同值--> 发布 speed_value
    |                          |--连续M帧未检出--> LOST --M*4 帧未检出--> 删除
    |--M帧未检出--> 删除(候选未确认即丢)
```

- **N=3** 连续检出→确认（对应 §8「连续 3 帧检出」）。
- **M=3** 连续未检出→lost（覆盖 `_1` 型低分漏检帧）。
- **K=3** 连续同值投票→发限速值（对应 §8「分类正确」+「连续≥3 同错值不通过」）。
- 关联用**中心距离/尺寸门控**（默认 `assoc_gate=2.0`，即中心移动 ≤ 2×目标尺寸），
  比 IoU 更适合小牌快速移动/放大（实测 _5 近距牌 IoU 会掉到 0.17~0.5，中心距离则稳）。

## 5. 构建与运行（板端）

```bash
source /opt/tros/humble/setup.bash
cd /home/sunrise/tsr_ws
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash

# 端到端（重放 _5 10Hz → 检测 → 跟踪 → 发布；调参在 YAML）
ros2 run tsr_node tsr_detect --det /home/sunrise/tsr_models/det_cropcolor1_352x640_bayese_640x352_nv12.bin \
                              --cls /home/sunrise/tsr_models/cls_v3_64x64_nv12_64x64.bin \
                              --ros-args --params-file src/tsr_node/config/tsr_node.yaml &
ros2 run tsr_node tsr_replay --dir /home/sunrise/tsr_test/self/images/test/20260907_5 --fps 10 &
ros2 topic echo speed_limit_signs

# 离线可视化（tsr_test --vis，画框+速度标签存 PNG）
./install/tsr_node/lib/tsr_node/tsr_test --det $DET --cls $CLS \
    --dir <trip_dir> --vis <out_dir>
```

调参文件 `config/tsr_node.yaml`（ROS2 参数）：
```yaml
tsr_node:
  ros__parameters:
    score_thres: 0.25   # 板端量化 conf 阈值（== 主机 fp32 0.5）
    N: 3                # 连续检出确认帧数
    M: 3                # 连续未检出判 lost 帧数
    K: 3                # 连续同值投票帧数
    assoc_gate: 2.0     # 中心距离关联门控（单位=目标尺寸）
```

## 6. 测试结论与注意事项

- **C++ 移植 = Python 逐框一致**：det 召回 69/185、FP 2、e2e 89.9%，与 `eval_board.py` 完全相同。
- **置信度偏移**：det 用 `score_thres=0.25`（等价主机 fp32 @0.5，见 `rdk_x5_deploy.md`）。
- **crop 路径**：节点输入是 NV12，crop 时**先按框切出小块 NV12 再转 BGR**（`nv12_crop_to_bgr`，
  用 `cv::COLOR_YUV2BGR_NV12` 只转 crop 区域），避免整帧转换；再 resize 64×64 → BGR→NV12。
  与训练/验证的 BGR crop 管线一致（保留 BGR 域 resize）。
- **帧率**：10Hz 重放（0.1s/帧）；tracker 参数 N/M/K 以 10Hz 为基准（若实际更高帧率需按时间折算）。
- **尚未做**：真机去畸变/正确 crop、`hobot_vio` 直采接入、多牌并发压力（_9 已测到 3 个生命周期）。

## 7. 交给同事集成前

- 输出消息 `SpeedLimitSign/SpeedLimitSignArray` 为占位 schema，可按集成方约定调整。
- 输入 `Nv12Image` 需与真实驱动对齐（或加适配层转 `hbm_img_msgs`/共享内存）。
- 可执行名：`ros2 run tsr_node tsr_detect`（不是 `tsr_node`，因与接口包同名）。
- 调参走 `config/tsr_node.yaml`（ROS2 参数）；`det/cls` 模型路径仍走 CLI `--det/--cls`。
