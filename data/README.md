# data/ — 数据目录

| 子目录 | 用途 | 状态 |
|--------|------|------|
| `TT100K/` | 公开 TT100K 全量 221 类（YOLO） | 已下载并转换 |
| `TT100K_speed_det/` | 限速检测子集（单类，仅 pl*+il* 框；images 软链到 TT100K） | 已生成 |
| `TT100K_speed_mc/` | 限速多类子集（**pl*+il*，29 类**；pm* 已移出） | 已生成 |
| `TT100K_weight_pm/` | **限重归档（pm*，非速度）**：hard-negative 备用，勿当正类训练 | 已生成 |
| `TT100K_speed_meta/` | `class_map.json` / `weight_class_map.json` / `filter_stats.json` / `color_flags.json`（信息性颜色标记，非错误清单） | 已生成 |
| `TT100K_speed_crops/` | sign-centric crops（pl*+il*，tight/1.3/1.5/2.0×，30936 crops）+ `manifest.csv` + `meta/crop_stats.json`；由 `scripts/crop_speed_signs.py` 生成，体检见 `reports/crop_stats.md` | 已重建（2026-09-23 语义修正后） |
| `self/` | 目标场景自采测试（GS130WI / RDK X5） | 空，预留 |

大文件 gitignore；本 README 可入库。

## TT100K

- 配置：`configs/TT100K.yaml`（`path` 为绝对路径）
- 下载脚本：`scripts/download_tt100k.py`
- 许可：CC BY-NC 2.0

期望划分：train 6105 / val 7641 / test 3071。

## self/（日后）

```
data/self/
└── raw/<trip_or_date>/   # 已去畸变优先；分辨率预期 640×352
```

标注格式与统一标签体系对齐后再写入 `docs/strategy.md`。
