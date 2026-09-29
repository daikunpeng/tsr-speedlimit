#!/usr/bin/env python3
"""Score GS130WI sign events under the revised acceptance criterion.

No model inference. GT boxes come from labels_det in the same order as
eval_e2e.py's per-image gt records; automatic physical-sign tracks are proposals
for human review, not new annotations.

Revised criterion (user 2026-09-24, docs/strategy.md §8.5):
  PASS   = within a target's lifecycle (>=min-lifecycle GT frames) there exist
           >=`run` CONSECUTIVE frames that are E2E-correct (det TP + cls correct).
           No pixel-size gate.
  FAIL   = verifiable lifecycle but no such run, OR a stable wrong value:
           >=`wrong-run` consecutive frames of the SAME wrong speed class.
  UNVERIFIABLE = lifecycle has < min-lifecycle GT frames (cannot reach a run).

Usage:
    .venv/bin/python scripts/eval_self_events.py
    .venv/bin/python scripts/eval_self_events.py --raw runs/eval/<name>/e2e_raw.json \
        --name <name> [--run 3 --wrong-run 3 --min-lifecycle 3]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def frame_index(key: str) -> int:
    try:
        return int(key.rsplit("_", 1)[1])
    except (ValueError, IndexError) as e:
        raise ValueError(f"cannot parse frame index from {key!r}") from e


def load_boxes(path: Path, width: int, height: int) -> list[list[float]]:
    boxes = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        parts = ln.split()
        if len(parts) != 5:
            raise ValueError(f"bad YOLO label in {path}: {ln!r}")
        _, cx, cy, w, h = map(float, parts)
        boxes.append([(cx - w / 2) * width, (cy - h / 2) * height,
                      (cx + w / 2) * width, (cy + h / 2) * height])
    return boxes


def iou(a: list[float], b: list[float]) -> float:
    x = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    y = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = x * y
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection + 1e-9)


def link_score(a: list[float], b: list[float], gap: int) -> float | None:
    """Tolerate camera movement and scale change; reject distant same-class signs."""
    overlap = iou(a, b)
    acx, acy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    size = max(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1])
    distance = math.hypot(acx - bcx, acy - bcy) / max(1.0, size)
    if overlap < 0.05 and distance > 0.65 * gap:
        return None
    return overlap - 0.1 * distance / gap


def build_tracks(views: list[dict], max_gap: int) -> list[list[dict]]:
    """Greedy one-to-one spatial association of GT boxes within each trip."""
    grouped: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for view in views:
        grouped[view["trip"]][view["frame"]].append(view)

    tracks: list[list[dict]] = []
    for trip, frames in sorted(grouped.items()):
        active: list[list[dict]] = []
        for idx in sorted(frames):
            current = frames[idx]
            candidates = []
            for vi, v in enumerate(current):
                for ti, track in enumerate(active):
                    last = track[-1]
                    if last["cls"] != v["cls"] or not 0 < idx - last["frame"] <= max_gap:
                        continue
                    score = link_score(last["xyxy"], v["xyxy"], idx - last["frame"])
                    if score is not None:
                        candidates.append((score, ti, vi))
            used_tracks: set[int] = set()
            used_views: set[int] = set()
            for _, ti, vi in sorted(candidates, reverse=True):
                if ti not in used_tracks and vi not in used_views:
                    active[ti].append(current[vi])
                    used_tracks.add(ti)
                    used_views.add(vi)
            for vi, v in enumerate(current):
                if vi not in used_views:
                    track = [v]
                    tracks.append(track)
                    active.append(track)
            active = [t for t in active if idx - t[-1]["frame"] <= max_gap]
    return tracks


def _max_run(seq: list[dict], is_target) -> list[dict]:
    """Longest consecutive run (frame index +1) satisfying is_target."""
    best, cur = [], []
    for o in seq:
        if is_target(o):
            cur.append(o)
        else:
            if len(cur) > len(best):
                best = cur
            cur = []
    if len(cur) > len(best):
        best = cur
    return best


def _wrong_run(seq: list[dict], cls: str) -> list[dict]:
    """Longest consecutive run of the SAME wrong speed class (not None/other/cls)."""
    best, cur = [], []
    last = None
    for o in seq:
        p = o["pred_cls"]
        if p not in (None, "other", cls):
            if cur and p == last and o["frame"] - cur[-1]["frame"] == 1:
                cur.append(o)
            else:
                if len(cur) > len(best):
                    best = cur
                cur = [o]
            last = p
        else:
            if len(cur) > len(best):
                best = cur
            cur = []
            last = None
    if len(cur) > len(best):
        best = cur
    return best


def judge_event(track: list[dict], run: int, wrong_run: int, min_lifecycle: int) -> dict:
    cls = track[0]["cls"]
    correct = _max_run(track, lambda o: o["pred_cls"] == cls)
    wrong = _wrong_run(track, cls)
    peak = max(track, key=lambda v: (v["size640"], v["frame"]))
    if len(track) < min_lifecycle:
        status = "unverifiable"
    elif len(wrong) >= wrong_run:
        status = "fail"  # stable wrong value
    elif len(correct) >= run:
        status = "pass"
    else:
        status = "fail"
    size_lo = min(o["size640"] for o in correct) if correct else None
    size_hi = max(o["size640"] for o in correct) if correct else None
    return {
        "trip": track[0]["trip"], "cls": cls,
        "start": track[0]["frame"], "end": track[-1]["frame"],
        "n_lifecycle_frames": len(track),
        "n_correct_frames": sum(o["pred_cls"] == cls for o in track),
        "peak_frame": peak["frame"], "max_size640": round(peak["size640"], 1),
        "correct_run_len": len(correct),
        "correct_run_frames": [o["frame"] for o in correct],
        "correct_run_size": (round(size_lo, 1), round(size_hi, 1)) if correct else None,
        "wrong_run_len": len(wrong),
        "wrong_run_class": wrong[0]["pred_cls"] if wrong else None,
        "status": status,
        "observations": [{"frame": o["frame"], "image": o["image"],
                          "size640": round(o["size640"], 2),
                          "pred_cls": o["pred_cls"]} for o in track],
    }


def _has_votes_in_span(views: list[dict], window: int, votes: int) -> bool:
    return any(views[i + votes - 1]["frame"] - views[i]["frame"] <= window - 1
               for i in range(len(views) - votes + 1))


def unmatched_speed_candidates(records: list[dict], max_gap: int, window: int,
                               votes: int, scale640: float) -> tuple[int, list[dict]]:
    """Propose spatially persistent *unmatched* speed-class predictions for review."""
    views = []
    for rec in records:
        key = rec["image"]
        trip = key.split("/", 1)[0]
        for p in rec["preds"]:
            if p["matched"] or p["cls"] in ("", "other"):
                continue
            box = p["xyxy"]
            views.append({"trip": trip, "frame": frame_index(key), "cls": p["cls"],
                          "xyxy": box, "image": key,
                          "size640": min(box[2] - box[0], box[3] - box[1]) * scale640})
    candidates = []
    for track in build_tracks(views, max_gap):
        if not _has_votes_in_span(track, window, votes):
            continue
        candidates.append({"trip": track[0]["trip"], "cls": track[0]["cls"],
                           "frames": [v["frame"] for v in track],
                           "max_pred_size640": round(max(v["size640"] for v in track), 1),
                           "sample_image": track[0]["image"],
                           "stable_after_12px_gate": _has_votes_in_span(
                               [v for v in track if v["size640"] >= 12], window, votes),
                           "stable_after_24px_gate": _has_votes_in_span(
                               [v for v in track if v["size640"] >= 24], window, votes)})
    return len(views), candidates


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="runs/eval/e2e_self/e2e_raw.json")
    ap.add_argument("--labels-root", default="data/self/labels_det/test")
    ap.add_argument("--name", default="self_baseline", help="unique output name per model/experiment")
    ap.add_argument("--run", type=int, default=3, help="min consecutive E2E-correct frames to pass")
    ap.add_argument("--wrong-run", type=int, default=3, help="stable wrong value length to fail")
    ap.add_argument("--min-lifecycle", type=int, default=3, help="min GT frames to be verifiable")
    ap.add_argument("--max-gap", type=int, default=5,
                    help="maximum frame-index gap when associating the same physical sign")
    args = ap.parse_args()
    if args.run < 1 or args.wrong_run < 1 or args.min_lifecycle < 1 or args.max_gap < 1:
        ap.error("run/wrong-run/min-lifecycle/max-gap must be >= 1")

    raw = json.loads((ROOT / args.raw).read_text(encoding="utf-8"))
    width, height = map(int, raw["src_size"].split("x"))
    labels_root = ROOT / args.labels_root
    views = []
    for rec in raw["records"]:
        key = rec["image"]
        if "/" not in key:
            raise ValueError(f"expected <trip>/<stem> key, got {key!r}")
        trip, _ = key.split("/", 1)
        boxes = load_boxes(labels_root / f"{key}.txt", width, height)
        if len(boxes) != len(rec["gt"]):
            raise ValueError(f"GT count mismatch for {key}: {len(boxes)} labels vs {len(rec['gt'])} E2E")
        for box, gt in zip(boxes, rec["gt"]):
            if not gt.get("gt_cls"):
                raise ValueError(f"missing speed class for {key}")
            views.append({"trip": trip, "frame": frame_index(key), "image": key,
                          "cls": gt["gt_cls"], "xyxy": box,
                          "size640": min(box[2] - box[0], box[3] - box[1]) * raw["scale_to_640"],
                          "pred_cls": (gt.get("pred_cls") or None) if gt.get("det_tp") else None})

    tracks = build_tracks(views, args.max_gap)
    events = [judge_event(track, args.run, args.wrong_run, args.min_lifecycle) for track in tracks]
    events.sort(key=lambda e: (e["trip"], e["start"], e["cls"]))
    for i, e in enumerate(events, 1):
        e["event_id"] = i
    counts = Counter(e["status"] for e in events)
    n_verifiable = counts["pass"] + counts["fail"]
    speed_fp, fp_candidates = unmatched_speed_candidates(
        raw["records"], args.max_gap, window=5, votes=3, scale640=raw["scale_to_640"])

    lines = [
        "# 自采数据：逐路牌事件评估（开发集）", "",
        f"- 生成：{datetime.now(timezone.utc).isoformat()}；E2E 明细：`{args.raw}`",
        f"- 通过标准：某生命周期内存在**连续 {args.run} 帧检出且分类正确**（无像素尺寸门槛）；"
        f"连续 ≥{args.wrong_run} 帧同一错误速度值为硬性不通过；"
        f"生命周期 <{args.min_lifecycle} 帧单列不可验",
        "- 事件由 GT 位置与帧间隔自动关联（**ID/边界需人工复核**）；检出=det conf≥0.5、"
        "匹配 IoU≥0.5；未匹配的速度类输出另作空间候选扫描，可能含漏标真牌。",
        "", "## 概览", "",
        f"- 总事件 **{len(events)}**；可验 **{n_verifiable}/{len(events)}**"
        f"（{n_verifiable / max(1, len(events)) * 100:.0f}%）；不可验 {counts['unverifiable']}",
        f"- **通过 {counts['pass']}/{n_verifiable}**"
        f"（{counts['pass'] / max(1, n_verifiable) * 100:.0f}%，按可验计；"
        f"占全部 {counts['pass']}/{len(events)}）",
        f"- 未匹配 GT 且被 cls 输出为速度值的检测框 {speed_fp}；"
        f"同值、任意 5 帧内 ≥3 次的候选轨迹 **{len(fp_candidates)}**"
        f"（≥12px 门控 {sum(c['stable_after_12px_gate'] for c in fp_candidates)} / "
        f"≥24px {sum(c['stable_after_24px_gate'] for c in fp_candidates)}）", "",
        "## 逐事件", "",
        "| ID | trip | GT | 生命周期帧 | 正确/总 | 最长连续正确 | 帧段 | 尺寸 | 最长错连/类 | 结果 |",
        "|---:|------|----|-----------:|---------:|------------:|------|------|--------------|------|",
    ]
    for e in events:
        span = f"{e['start']}–{e['end']}"
        run_txt = (f"{e['correct_run_len']} | "
                   f"{e['correct_run_frames'][0]}..{e['correct_run_frames'][-1]}"
                   if e["correct_run_len"] else "0 | —")
        size_txt = (f"{e['correct_run_size'][0]:.0f}–{e['correct_run_size'][1]:.0f}px"
                    if e["correct_run_size"] else "—")
        wrong = f"{e['wrong_run_len']}/{e['wrong_run_class'] or '—'}"
        lines.append(f"| {e['event_id']} | {e['trip']} | {e['cls']} | {e['n_lifecycle_frames']} | "
                     f"{e['n_correct_frames']}/{e['n_lifecycle_frames']} | {run_txt} | {size_txt} | "
                     f"{wrong} | {e['status']} |")
    lines += ["", "## 未匹配速度值的稳定候选（仅供复核）", "",
              "| trip | 输出 | 帧号 | 最大框短边 | 示例图 |",
              "|------|------|------|------------:|--------|"]
    for c in fp_candidates:
        frames = ",".join(map(str, c["frames"]))
        lines.append(f"| {c['trip']} | {c['cls']} | {frames} | "
                     f"{c['max_pred_size640']:.1f}px | `{c['sample_image']}` |")
    if not fp_candidates:
        lines.append("| — | — | — | — | — |")
    lines += ["", f"原始事件与逐帧明细：`runs/eval/{args.name}/event_eval_raw.json`", ""]
    out = ROOT / "runs" / "eval" / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "event_eval_raw.json").write_text(json.dumps({
        "raw": args.raw, "run": args.run, "wrong_run": args.wrong_run,
        "min_lifecycle": args.min_lifecycle, "max_gap": args.max_gap, "events": events,
        "n_unmatched_speed_predictions": speed_fp, "unmatched_stable_candidates": fp_candidates,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    report = ROOT / "reports" / f"event_eval_{args.name}.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"events {len(events)} | verifiable {n_verifiable} | pass {counts['pass']} | "
          f"fail {counts['fail']} | unverifiable {counts['unverifiable']} | wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
