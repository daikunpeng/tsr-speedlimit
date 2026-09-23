# 开发环境安装与 TT100K 下载

采集/安装日期：2026-09-22

## 一键命令（本机已验证）

```bash
cd /home/sicv/ws/tsr_speedlimit

# 环境冒烟（直接用 venv，避免 uv run 重新解析 torch）
.venv/bin/python scripts/check_env.py

# 下载 TT100K 到 data/TT100K（约 17GB zip + 转换，可续传）
.venv/bin/python scripts/download_tt100k.py --dry-run   # 只看路径与磁盘
.venv/bin/python scripts/download_tt100k.py

# 限速类别过滤 → data/TT100K_speed_det + data/TT100K_speed_mc
.venv/bin/python scripts/filter_speed_limit.py --dry-run
.venv/bin/python scripts/filter_speed_limit.py --force
```

## 环境概要

| 项 | 版本/路径 |
|----|-----------|
| Python | 3.11.13（uv `.venv`） |
| torch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| ultralytics | 8.4.x |
| 权重冒烟 | `weights/yolov5nu.pt`（Ultralytics YOLOv5u nano） |
| 数据 | `data/TT100K/`（绝对路径写在 `configs/TT100K.yaml`） |

硬件细节见 [local_environment.md](local_environment.md)。

## 依赖安装说明（重要）

本机直连 `pypi.nvidia.com` / 官方 PyTorch CDN 不稳定。实际采用：

1. 从 **阿里云 PyTorch 镜像** 下载 `torch` / `torchvision` cu124 轮子到 `.wheels/`
2. 从 **阿里云 PyPI** 拉取 `nvidia-*-cu12` / `triton` 等到 `.wheels/`
3. `uv pip install --find-links .wheels --index-url https://mirrors.aliyun.com/pypi/simple ...`
4. 再 `uv pip install ultralytics ...`（同样用阿里云 simple）

不要用裸 `uv add torch --index https://download.pytorch.org/whl/cu124`（易超时）。  
不要用 `uv run` 跑脚本，除非已确认不会触发重新下载 CPU/错误版 torch；统一：

```bash
.venv/bin/python scripts/xxx.py
```

`.wheels/` 已 gitignore，可在装好后删除以省磁盘（约数 GB）。

## 代理（中国 + 科学上网）

本机常开科学上网（示例端口 `127.0.0.1:7890`）。**有代理 ≠ 所有下载都该走代理**；按目标选路径更稳、更快。

### 推荐用法

| 场景 | 建议 | 原因（本机实测） |
|------|------|------------------|
| 阿里云 / 清华等**国内镜像**（PyTorch wheel、PyPI、TT100K） | **关代理**再下 | 直连常 5–10 MB/s；硬走代理反而慢或超时 |
| GitHub releases（如 `yolov5nu.pt`）、部分海外 CDN | 只用 **HTTP 代理** | `http_proxy=http://127.0.0.1:7890` |
| `uv` / 部分 Python HTTP 库 | **关掉 SOCKS 总代理** | `ALL_PROXY=socks://...` 常导致超时、TLS 中断 |

```bash
# 国内镜像 / TT100K（推荐）
unset ALL_PROXY all_proxy http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# 仅访问 GitHub 等海外源时
unset ALL_PROXY all_proxy
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
```

### 不要这样

- 不要依赖 shell 里默认的 `ALL_PROXY=socks://127.0.0.1:7890/` 跑 `uv add` / 大包下载（本机曾因此在 `pypi.nvidia.com` 反复失败）。
- 不要假设「科学上网开着，官方 PyTorch CUDA 索引就一定比阿里云镜像快」——CUDA 大轮子优先走阿里云镜像更省事。

### Ultralytics 训练首次联网（yolo26n / Arial）

- 训练启动时 `check_amp()` 会一次性下载 `weights/yolo26n.pt` **仅作 AMP 自检**（日志：`not used for training`），
  与部署模型 `yolov5nu.pt` 无关，**不会上 X5 板**；`weights/*.pt` 已 gitignore。
- GitHub 直连/代理都可能卡死在个位数百分比。已下好的话跳过；若需重下可用国内镜像前缀：
  `curl -L -o weights/yolo26n.pt "https://gh-proxy.com/https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt"`
- 字体 `~/.config/Ultralytics/Arial.ttf` 同理（plot 用，一次性）。

## 数据布局

```
data/
├── README.md
├── TT100K/          # 公开集（本阶段）
│   ├── data.zip     # 原始包（转换后可删以省空间）
│   ├── data/        # 解压原始（可选）
│   ├── images/{train,val,test}
│   └── labels/{train,val,test}
└── self/            # 自采测试集（预留，现空）
```

期望图像数：train 6105 / val 7641 / test 3071。

## 验收

- [x] `scripts/check_env.py` → CUDA + yolov5nu **PASS**
- [x] `scripts/download_tt100k.py` → **VERIFY OK**
  - train images ≈ 6103（文档 6105）
  - val images = 7641（无目标图可不生成 label 文件，故 labels 数会少于 images）
  - test images ≈ 3067（文档 3071）
  - 落盘：`data/TT100K/`（含 `data.zip` + `data/` 原始解压 + `images/` + `labels/`，合计约数十 GB）

## 限速过滤产物

| 项 | 路径 |
|----|------|
| 检测单类 | `configs/TT100K_speed_det.yaml` → `data/TT100K_speed_det/` |
| 多类限速 | `configs/TT100K_speed.yaml` → `data/TT100K_speed_mc/` |
| 映射 / 统计 | `data/TT100K_speed_meta/`、`reports/speed_filter_stats.md` |

图像经符号链接复用 `data/TT100K/images`，不重复占磁盘。
