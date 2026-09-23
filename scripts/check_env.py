#!/usr/bin/env python3
"""Smoke-check project env: torch CUDA, ultralytics, yolov5n load."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = ROOT / "weights"
REPORTS_DIR = ROOT / "reports"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-weights",
        action="store_true",
        help="Skip downloading/loading yolov5n.pt",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPORTS_DIR / "env_smoke.md",
        help="Write markdown report path",
    )
    args = parser.parse_args()

    lines: list[str] = []
    lines.append(f"# Env smoke — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")

    import torch

    lines.append(f"- Python: `{sys.version.split()[0]}`")
    lines.append(f"- torch: `{torch.__version__}`")
    lines.append(f"- cuda available: `{torch.cuda.is_available()}`")
    if not torch.cuda.is_available():
        text = "\n".join(lines + ["", "**FAIL**: CUDA not available"])
        _write(args.report, text)
        print(text)
        return 1

    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    lines.append(f"- GPU: `{name}` ({total:.1f} GiB)")

    a = torch.randn(1024, 1024, device="cuda")
    b = torch.randn(1024, 1024, device="cuda")
    c = a @ b
    torch.cuda.synchronize()
    lines.append(f"- CUDA matmul ok: shape `{tuple(c.shape)}`")

    import ultralytics

    lines.append(f"- ultralytics: `{ultralytics.__version__}`")

    if not args.skip_weights:
        from ultralytics import YOLO

        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        # Ultralytics recommends yolov5nu.pt (YOLOv5u trained in this repo)
        weight = WEIGHTS_DIR / "yolov5nu.pt"
        model_name = str(weight) if weight.exists() else "yolov5nu.pt"
        model = YOLO(model_name)
        if not weight.exists():
            try:
                import shutil

                for cand in (
                    getattr(model, "ckpt_path", None),
                    ROOT / "yolov5nu.pt",
                    Path("yolov5nu.pt"),
                ):
                    if cand and Path(cand).exists():
                        shutil.copy2(cand, weight)
                        break
            except Exception as exc:  # noqa: BLE001
                lines.append(f"- weight copy warning: `{exc}`")
        lines.append(f"- yolov5nu loaded: `{weight if weight.exists() else model_name}`")

    lines.append("")
    lines.append("**PASS**")
    text = "\n".join(lines)
    _write(args.report, text)
    print(text)
    return 0


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
