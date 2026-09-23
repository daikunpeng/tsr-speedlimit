# TT100K speed-limit filter stats（语义修正版）

- Generated: `2026-09-23T01:51:39.394783+00:00`
- Source: `/home/sicv/ws/tsr_speedlimit/data/TT100K`
- Speed multiclass out: `/home/sicv/ws/tsr_speedlimit/data/TT100K_speed_mc`（pl*+il*，29 类）
- Detector out: `/home/sicv/ws/tsr_speedlimit/data/TT100K_speed_det`（单类 speed_limit_sign）
- Weight archive out: `/home/sicv/ws/tsr_speedlimit/data/TT100K_weight_pm`（pm*，17 类，**非速度，不参与训练**）
- 语义：pl=最高限速、il=最低限速（蓝底）、pm=限重（吨）；plx/pmx/ilx 无实例，未纳入（见 reports/label_audit.md）

## Speed classes

| new_id | old_id | name | group | semantic | value |
|-------:|-------:|------|-------|----------|------:|
| 0 | 11 | `il100` | il | speed_min | 100.0 |
| 1 | 12 | `il110` | il | speed_min | 110.0 |
| 2 | 13 | `il50` | il | speed_min | 50.0 |
| 3 | 14 | `il60` | il | speed_min | 60.0 |
| 4 | 15 | `il70` | il | speed_min | 70.0 |
| 5 | 16 | `il80` | il | speed_min | 80.0 |
| 6 | 17 | `il90` | il | speed_min | 90.0 |
| 7 | 175 | `ilx` | il | speed_min |  |
| 8 | 213 | `pl0` | pl | speed_max | 0.0 |
| 9 | 76 | `pl10` | pl | speed_max | 10.0 |
| 10 | 77 | `pl100` | pl | speed_max | 100.0 |
| 11 | 78 | `pl110` | pl | speed_max | 110.0 |
| 12 | 79 | `pl120` | pl | speed_max | 120.0 |
| 13 | 80 | `pl15` | pl | speed_max | 15.0 |
| 14 | 81 | `pl20` | pl | speed_max | 20.0 |
| 15 | 82 | `pl25` | pl | speed_max | 25.0 |
| 16 | 215 | `pl3` | pl | speed_max | 3.0 |
| 17 | 83 | `pl30` | pl | speed_max | 30.0 |
| 18 | 84 | `pl35` | pl | speed_max | 35.0 |
| 19 | 214 | `pl4` | pl | speed_max | 4.0 |
| 20 | 85 | `pl40` | pl | speed_max | 40.0 |
| 21 | 86 | `pl5` | pl | speed_max | 5.0 |
| 22 | 87 | `pl50` | pl | speed_max | 50.0 |
| 23 | 88 | `pl60` | pl | speed_max | 60.0 |
| 24 | 89 | `pl65` | pl | speed_max | 65.0 |
| 25 | 90 | `pl70` | pl | speed_max | 70.0 |
| 26 | 91 | `pl80` | pl | speed_max | 80.0 |
| 27 | 92 | `pl90` | pl | speed_max | 90.0 |
| 28 | 198 | `plx` | pl | speed_max |  |

## Weight archive classes（留档：hard-negative 备用）

| new_id | old_id | name | value (t) |
|-------:|-------:|------|----------:|
| 0 | 96 | `pm1.5` | 1.5 |
| 1 | 93 | `pm10` | 10.0 |
| 2 | 94 | `pm13` | 13.0 |
| 3 | 95 | `pm15` | 15.0 |
| 4 | 97 | `pm2` | 2.0 |
| 5 | 216 | `pm2.5` | 2.5 |
| 6 | 98 | `pm20` | 20.0 |
| 7 | 99 | `pm25` | 25.0 |
| 8 | 100 | `pm30` | 30.0 |
| 9 | 101 | `pm35` | 35.0 |
| 10 | 102 | `pm40` | 40.0 |
| 11 | 103 | `pm46` | 46.0 |
| 12 | 104 | `pm5` | 5.0 |
| 13 | 105 | `pm50` | 50.0 |
| 14 | 106 | `pm55` | 55.0 |
| 15 | 107 | `pm8` | 8.0 |
| 16 | 199 | `pmx` | None |

## Splits

### train

- images: 6103
- label files written: 6103
- images with ≥1 speed box: 3393
- empty negatives: 2710
- speed boxes kept: 5002（weight 归档 317，排除 0，其余类丢弃 11205）
- bbox width px percentiles: `{'p50': 39.000064, 'p75': 60.000256, 'p90': 88.999936, 'p95': 111.9985664}` (n=5002)
- bbox height px percentiles: `{'p50': 43.200512, 'p75': 64.94976, 'p90': 91.199488, 'p95': 110.665728}`

top speed classes:

| name | count |
|------|------:|
| `pl40` | 874 |
| `pl50` | 660 |
| `pl80` | 578 |
| `pl60` | 526 |
| `pl100` | 449 |
| `pl30` | 373 |
| `il60` | 338 |
| `pl5` | 269 |
| `pl120` | 208 |
| `il80` | 197 |
| `pl70` | 103 |
| `pl20` | 98 |
| `il100` | 92 |
| `il90` | 59 |
| `pl90` | 42 |

top weight classes:

| name | count |
|------|------:|
| `pm20` | 107 |
| `pm55` | 98 |
| `pm30` | 75 |
| `pm10` | 13 |
| `pm15` | 7 |
| `pm35` | 4 |
| `pm2` | 4 |
| `pm40` | 3 |
| `pm5` | 2 |
| `pm50` | 2 |

### val

- images: 7641
- label files written: 1097
- images with ≥1 speed box: 203
- empty negatives: 894
- speed boxes kept: 242（weight 归档 100，排除 0，其余类丢弃 1302）
- bbox width px percentiles: `{'p50': 15.034368, 'p75': 37.248000000000005, 'p90': 78.43082240000001, 'p95': 93.94534399999995}` (n=242)
- bbox height px percentiles: `{'p50': 17.067008, 'p75': 47.842816, 'p90': 86.28817920000002, 'p95': 109.06583039999994}`

top speed classes:

| name | count |
|------|------:|
| `pl15` | 41 |
| `pl50` | 31 |
| `pl40` | 27 |
| `pl5` | 26 |
| `pl80` | 23 |
| `pl35` | 17 |
| `pl110` | 13 |
| `pl10` | 11 |
| `pl30` | 10 |
| `pl60` | 10 |
| `pl90` | 10 |
| `pl100` | 6 |
| `il60` | 5 |
| `il80` | 2 |
| `pl0` | 2 |

top weight classes:

| name | count |
|------|------:|
| `pm10` | 34 |
| `pm15` | 18 |
| `pm5` | 11 |
| `pm25` | 9 |
| `pm50` | 7 |
| `pm8` | 7 |
| `pm46` | 6 |
| `pm40` | 4 |
| `pm1.5` | 2 |
| `pm13` | 1 |

### test

- images: 3067
- label files written: 3067
- images with ≥1 speed box: 1735
- empty negatives: 1332
- speed boxes kept: 2490（weight 归档 135，排除 0，其余类丢弃 5556）
- bbox width px percentiles: `{'p50': 36.09088, 'p75': 57.99936, 'p90': 89.11790079999999, 'p95': 112.44124159999978}` (n=2490)
- bbox height px percentiles: `{'p50': 41.332736, 'p75': 65.333248, 'p90': 94.70013439999998, 'p95': 112.63743999999969}`

top speed classes:

| name | count |
|------|------:|
| `pl40` | 445 |
| `pl50` | 340 |
| `pl60` | 274 |
| `pl80` | 274 |
| `pl100` | 216 |
| `pl30` | 205 |
| `pl5` | 203 |
| `il60` | 140 |
| `il80` | 96 |
| `pl120` | 87 |
| `pl20` | 56 |
| `pl70` | 44 |
| `il100` | 39 |
| `il90` | 17 |
| `pl15` | 17 |

top weight classes:

| name | count |
|------|------:|
| `pm20` | 49 |
| `pm55` | 38 |
| `pm30` | 32 |
| `pm15` | 5 |
| `pm2` | 2 |
| `pm5` | 2 |
| `pm50` | 2 |
| `pm2.5` | 1 |
| `pm10` | 1 |
| `pm8` | 1 |

