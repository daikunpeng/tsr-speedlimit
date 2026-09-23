#!/usr/bin/env python3
"""Download TT100K into <repo>/data/TT100K and convert to YOLO format."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TT100K_DIR = DATA_DIR / "TT100K"
CONFIG = ROOT / "configs" / "TT100K.yaml"
MIN_FREE_GB = 40.0
EXPECTED = {"train": 6105, "val": 7641, "test": 3071}
ZIP_URL = "https://cg.cs.tsinghua.edu.cn/traffic-sign/data_model_code/data.zip"


def free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024**3)


def already_converted() -> bool:
    for split in EXPECTED:
        img_dir = TT100K_DIR / "images" / split
        if not img_dir.is_dir():
            return False
        n = sum(1 for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if n < EXPECTED[split] * 0.95:
            return False
    return True


def ensure_zip() -> Path:
    """Download data.zip into TT100K_DIR with curl resume support."""
    TT100K_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = TT100K_DIR / "data.zip"

    # Adopt any ultralytics-style partial downloads
    for part in TT100K_DIR.glob(".data.zip.*.part"):
        if not zip_path.exists() or part.stat().st_size > zip_path.stat().st_size:
            print(f"adopting partial: {part.name} -> data.zip")
            part.replace(zip_path)

    if zip_path.exists() and zip_path.stat().st_size > 15 * (1024**3):
        print(f"zip already present: {zip_path} ({zip_path.stat().st_size / 1e9:.2f} GB)")
        return zip_path

    size = zip_path.stat().st_size if zip_path.exists() else 0
    print(f"curl download -> {zip_path} (resume from {size / 1e6:.1f} MB)")
    cmd = [
        "curl",
        "-L",
        "--retry",
        "20",
        "--retry-delay",
        "3",
        "--retry-all-errors",
        "-C",
        "-",
        "--connect-timeout",
        "30",
        "-o",
        str(zip_path),
        ZIP_URL,
    ]
    subprocess.run(cmd, check=True)
    print(f"zip done: {zip_path.stat().st_size / 1e9:.2f} GB")
    return zip_path


def verify() -> None:
    print("Verifying splits...")
    for split, expect in EXPECTED.items():
        img_dir = TT100K_DIR / "images" / split
        label_dir = TT100K_DIR / "labels" / split
        n_img = sum(1 for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        n_lbl = sum(1 for p in label_dir.iterdir() if p.suffix == ".txt") if label_dir.is_dir() else 0
        print(f"  {split}: images={n_img} (expect~{expect}), labels={n_lbl}")
        if n_img < expect * 0.95:
            raise SystemExit(f"FAIL: {split} image count too low ({n_img} < {expect})")

    sample = next((TT100K_DIR / "images" / "train").glob("*.jpg"))
    label = TT100K_DIR / "labels" / "train" / f"{sample.stem}.txt"
    from PIL import Image

    with Image.open(sample) as im:
        print(f"  sample image: {sample.name} size={im.size}")
    if label.exists():
        first = label.read_text(encoding="utf-8").strip().splitlines()[:1]
        print(f"  sample label line: {first[0] if first else '(empty)'}")
    else:
        print("  sample label: missing (image may have no objects)")
    print("VERIFY OK")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Only print paths and disk space")
    parser.add_argument("--force", action="store_true", help="Re-download/convert even if present")
    args = parser.parse_args()

    if not CONFIG.exists():
        print(f"Missing config: {CONFIG}", file=sys.stderr)
        return 1

    free = free_gb(ROOT)
    print(f"repo: {ROOT}")
    print(f"data dir: {DATA_DIR}")
    print(f"TT100K dir: {TT100K_DIR}")
    print(f"config: {CONFIG}")
    print(f"free disk: {free:.1f} GiB (need >= {MIN_FREE_GB})")

    if free < MIN_FREE_GB:
        print("FAIL: not enough free disk", file=sys.stderr)
        return 1

    if args.dry_run:
        print("dry-run: skip download")
        return 0

    if already_converted() and not args.force:
        print("TT100K already converted; skipping download. Use --force to redo.")
        verify()
        return 0

    # 1) Ensure zip lands under project data/TT100K (absolute path in yaml)
    ensure_zip()

    # 2) Let Ultralytics convert (download step should no-op / reuse local zip)
    from ultralytics.utils import SETTINGS

    SETTINGS.update(
        {
            "datasets_dir": str(DATA_DIR),
            "weights_dir": str(ROOT / "weights"),
            "runs_dir": str(ROOT / "runs"),
        }
    )
    print(f"ultralytics datasets_dir -> {DATA_DIR}")

    from ultralytics.data.utils import check_det_dataset

    data = check_det_dataset(str(CONFIG))
    print(f"dataset ready: {data.get('path', TT100K_DIR)}")
    verify()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
