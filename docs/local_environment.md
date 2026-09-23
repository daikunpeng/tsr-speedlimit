# 本机环境调研

采集时间：2026-09-22  
主机：`sicv-Precision-3650-Tower`（Dell Precision 3650 Tower）  
用途：为 TSR 限速检测项目规划依赖、训练分辨率、`batch` 与磁盘预算提供依据。

复现命令见文末「采集方法」。硬件变更后应重跑并更新本文。

---

## 1. 结论（给本项目用）

| 项 | 现状 | 建议 |
|----|------|------|
| GPU | RTX 3060 **12 GB**，桌面已占用约 **0.7 GB** | 可用显存约 **&lt;11 GB**；先 `imgsz=640`，提高分辨率时减小 `batch` |
| CUDA | 驱动报 CUDA **13.2**；本机 toolkit/`nvcc` 为 **12.4** | PyTorch / Ultralytics 优先装 **cu12x** 轮子；不要升级 NVIDIA 驱动（桌面会话依赖当前 GPU） |
| 磁盘 | 系统盘可用约 **538 GB** | TT100K ~18 GB 足够；数据集放 `datasets/`，勿进 git |
| Python | 系统默认 **3.10.12**；uv 已装 **3.11.13** | 项目用 **uv + Python 3.11**，不用系统 Python |
| 代理 | 科学上网在用（`7890`） | **国内镜像关代理**；海外源只用 `http://` 代理；务必 `unset ALL_PROXY`（`socks://` 会坑 uv/httpx）。详见 `docs/setup.md` |

系统 Python **未安装** `torch`；本仓库环境尚未用 uv 初始化（本文档撰写时）。

---

## 2. 操作系统与会话

| 项 | 值 |
|----|-----|
| OS | Ubuntu 22.04.5 LTS (Jammy) |
| Kernel | 6.8.0-138-generic |
| Arch | x86_64 |
| 用户 | `sicv`（含 `sudo`、`docker`、`ollama` 组） |
| 桌面 | GNOME Shell 42.9，`XDG_SESSION_TYPE=x11`，`DISPLAY=:1` |
| 厂商 / 机型 | Dell Inc. / Precision 3650 Tower |

采集时已开机约 3 天；load average 约 0.8 / 0.5 / 0.4。

---

## 3. CPU

| 项 | 值 |
|----|-----|
| 型号 | Intel Core i7-11700 @ 2.50GHz（11th Gen，Rocket Lake） |
| 物理核 / 逻辑核 | 8 核 / 16 线程 |
| 频率 | 800–4900 MHz |
| 缓存 | L3 16 MiB |
| 虚拟化 | VT-x |
| 指令集亮点 | AVX2、AVX-512、AES-NI |

数据加载与 CPU 增强可适当提高 `workers`（常见 4–8），以磁盘与内存为准。

---

## 4. 内存

| 项 | 值 |
|----|-----|
| 物理内存 | **32 GB**（采集时 available ≈ 22 GB） |
| Swap | 2 GB |

训练 + 桌面 + 浏览器通常够用；大 `imgsz` / 大 batch 时优先看 **GPU 显存**，不是系统 RAM。

---

## 5. GPU 与 CUDA

### 5.1 设备

| 项 | 值 |
|----|-----|
| 独显 | NVIDIA GeForce RTX 3060（GA106，Lite Hash Rate） |
| 显存 | **12288 MiB（12 GB）** |
| Compute Capability | **8.6**（Ampere） |
| 采集时占用 | ~717 MiB / 12288 MiB（约 28% util，P8 低功耗） |
| 核显 | Intel UHD Graphics 750（RocketLake-S GT1，Display controller） |

`nvidia-smi` 可见桌面相关进程占用显存（约数）：Xorg、gnome-shell、Cursor、浏览器、远程/协作软件等。训练前可关多余 GPU 图形进程以腾出显存。

### 5.2 驱动与工具链（注意版本差）

| 项 | 版本 | 说明 |
|----|------|------|
| NVIDIA Driver | **595.91.07** | `nvidia-smi` 显示的驱动 |
| Driver 宣称 CUDA | **13.2** | 驱动能跑的最高 CUDA 用户态能力 |
| CUDA Toolkit（本机） | **12.4.1**（`/usr/local/cuda` → 12.4） | 编译自定义 CUDA 扩展时用 |
| `nvcc` | **12.4.131** | 与 toolkit 一致 |

**要点**：驱动 CUDA 13.2 ≠ 本机已装 toolkit 12.4。装 PyTorch 时选与 **cu12x** 匹配的预编译包即可；一般不必装 CUDA 13 toolkit。若编译需 CUDA 的扩展，令 `CUDA_HOME` 指向 12.4，**不要**为对齐而升级驱动。

---

## 6. 存储

| 设备 | 容量 | 文件系统 | 挂载 | 用途 |
|------|------|----------|------|------|
| KINGSTON SA2000M8 1TB NVMe | ~916G 可用分区 | ext4 | `/` | 系统与工程盘；可用约 **538 GB** |
| EFI | 512M | vfat | `/boot/efi` | 启动 |
| USB HDD | 931.5G | ntfs | `/media/sicv/E300_AND_XIAOBA` | 外置盘（勿默认当训练数据盘） |

TT100K（约 18 GB）放在项目 `datasets/` 下空间充足；仍建议预留训练产物与多次实验缓存（`runs/`、权重）。

---

## 7. 软件环境

### 7.1 语言与包管理

| 工具 | 版本 / 路径 | 备注 |
|------|-------------|------|
| 系统 Python | 3.10.12 → `/usr/bin/python3` | 另有 `/usr/bin/python3.12` |
| uv | **0.11.12** → `~/.local/bin/uv` | 本项目强制用 uv |
| uv 已缓存 Python | **3.11.13**（`~/.local/share/uv/python/...`） | 推荐项目解释器 |
| pip | 25.3（绑在用户 Python 3.12 site-packages） | 项目内勿依赖此全局 pip |
| conda | 未安装 | — |

系统 `python3`：**无** `torch`；有系统级 OpenCV Python 绑定 **4.5.4**（与项目 venv 无关，训练以 venv 为准）。

### 7.2 构建与多媒体

| 工具 | 版本 |
|------|------|
| gcc | 11.4.0 |
| clang | 未安装 |
| ffmpeg | 4.4.2（Ubuntu 包） |
| git | 2.34.1 |
| bash | 5.1.16 |

### 7.3 其它常用软件

| 工具 | 版本 |
|------|------|
| Node.js / npm | v24.11.0 / 11.6.1 |
| Docker | 28.4.0 |
| Ollama | 0.6.0 |
| Cursor | 3.18.25（x64） |

### 7.4 网络代理（环境变量）

本机 shell 常见配置（值已脱敏）：

- `http(s)_proxy` / `HTTP(S)_PROXY` → `http://127.0.0.1:7890/`
- `all_proxy` / `ALL_PROXY` → `socks://127.0.0.1:7890/`
- `no_proxy` → localhost / 本机回环

已知风险与建议（中国 + 科学上网）：

- `ALL_PROXY=socks://...`：uv / 部分库不认或易 TLS 中断 → 大包安装前先 `unset ALL_PROXY all_proxy`
- 国内镜像（阿里云 PyPI / PyTorch、清华 TT100K）：**直连通常更快**，不要强行走代理
- 海外源（GitHub assets 等）：用 `http(s)_proxy=http://127.0.0.1:7890` 即可
- 完整操作表见 [`setup.md`](setup.md)「代理」一节

---

## 8. 对本项目的训练参数建议（初值）

在 **未** 关桌面、显存约 11 GB 可用的前提下：

| 场景 | 建议初值 |
|------|----------|
| Smoke / 首次训练 | `yolo11n` 或同类 nano；`imgsz=640`；`batch=8`（OOM 则降到 4） |
| 小目标更好 | `imgsz=1280`；`batch=2–4` |
| DataLoader | `workers=4–8`（视 CPU/磁盘） |
| 精度 | 优先默认混合精度（Ultralytics 通常已开 AMP） |

全量 TT100K 下载与长训前须用户确认（见 `AGENTS.md`）。

---

## 9. 采集方法

以下命令在本机一次性采集（2026-09-22）：

```bash
uname -a; cat /etc/os-release
lscpu; free -h; df -h; lsblk
lspci | grep -iE 'vga|3d|display'
nvidia-smi; nvcc -V; cat /usr/local/cuda/version.json
python3 --version; uv --version; uv python list
gcc --version; ffmpeg -version | head -1
git --version; docker --version; node --version
env | grep -i proxy   # 仅看 scheme / 主机，勿把完整凭据写入文档
```

---

## 10. 变更记录

| 日期 | 说明 |
|------|------|
| 2026-09-22 | 初版：OS / CPU / 内存 / GPU / CUDA / 磁盘 / 软件栈 / 代理 / 训练建议 |
| 2026-09-22 | 项目 venv 已建（Python 3.11 + torch 2.6.0+cu124 + ultralytics）；TT100K 落盘 `data/TT100K`。安装细节见 `docs/setup.md` |
