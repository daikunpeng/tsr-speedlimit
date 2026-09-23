# Tiny classifier smoke 评估

- 生成：2026-09-23T04:49:33.007243+00:00；输入 scale=1.5 crops，64×64；参数 245k
- 训练 30 ep（train 5244 含并入的 val；**test 2490 独立评估**）
- **test 总准确率：94.86%**（best: 94.86%）

## 尺寸分桶准确率（GT crop，@640 等效 px）

| 桶 | n | acc |
|----|---|----:|
| <8 | 708 | 93.6 |
| 8-12 | 628 | 96.7 |
| 12-16 | 415 | 96.4 |
| 16-24 | 392 | 94.1 |
| 24-32 | 213 | 93.9 |
| 32-48 | 108 | 92.6 |
| >=48 | 26 | 88.5 |

## Top-20 混淆对

| true | pred | n |
|------|------|---:|
| il80 | il60 | 22 |
| pl40 | pl60 | 8 |
| pl80 | pl60 | 8 |
| il90 | il60 | 6 |
| pl40 | pl50 | 6 |
| pl70 | pl50 | 6 |
| il70 | il60 | 5 |
| il100 | il60 | 5 |
| il50 | il60 | 5 |
| pl10 | pl30 | 5 |
| pl60 | pl50 | 4 |
| pl50 | pl30 | 3 |
| pl30 | pl50 | 3 |
| pl110 | pl100 | 3 |
| il80 | il90 | 2 |
| pl5 | pl50 | 2 |
| pl5 | pl40 | 2 |
| pl100 | pl80 | 2 |
| pl80 | pl50 | 2 |
| pl50 | pl60 | 2 |

中间数据：`runs/classifier/smoke1/`（history / test_predictions / summary）
图：`reports/figures/cls_confusion.png`、`reports/figures/cls_size_acc.png`

