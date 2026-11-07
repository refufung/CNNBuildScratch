"""Build the split CSVs from the pre-split `input/Cleaned Dataset` folders.

The input dataset already ships official train/validation/test folders
(700/150/150 images per class), so no re-splitting happens here — there is no
train<->test leakage risk to guard against. This script simply:

  1. indexes the images into outputs/{train,val,test}.csv (via dataset.ensure_split_csvs)
  2. computes train mean/std into outputs/split_meta.json
  3. verifies the result: per-class counts and zero cross-split path overlap

Existing CSVs are reused when valid; CSVs whose paths no longer exist (e.g.
written on another machine) are backed up to outputs/backup_prev/ and rebuilt.

Usage:
  python prepare_split.py           # build only when missing/stale
  python prepare_split.py --force   # back up and rebuild from input/ anyway
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from dataset import OUTPUT_DIR, ensure_split_csvs

SPLIT_NAMES = ("train", "val", "test")


def backup_existing() -> None:
    existing = [OUTPUT_DIR / f"{s}.csv" for s in SPLIT_NAMES]
    existing.append(OUTPUT_DIR / "split_meta.json")
    existing = [p for p in existing if p.is_file()]
    if not existing:
        return
    backup_dir = OUTPUT_DIR / "backup_prev"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for p in existing:
        shutil.move(str(p), backup_dir / p.name)
    print(f"Moved {len(existing)} existing file(s) to {backup_dir}")


def verify() -> bool:
    meta = json.loads((OUTPUT_DIR / "split_meta.json").read_text(encoding="utf-8"))
    print("\n=== verification ===")
    print(json.dumps(meta.get("class_counts", {}), indent=2))
    seen: set[str] = set()
    ok = True
    for name in SPLIT_NAMES:
        df = pd.read_csv(OUTPUT_DIR / f"{name}.csv")
        overlap = seen.intersection(df["path"])
        print(
            f"{name}: {len(df)} rows, labels={sorted(df['label'].unique().tolist())}, "
            f"overlap-with-previous-splits={len(overlap)}"
        )
        if len(overlap) or len(df) != df["path"].nunique():
            ok = False
        seen.update(df["path"])
    print("OK" if ok else "VERIFICATION FAILED")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="back up and rebuild CSVs even when valid")
    args = parser.parse_args()

    print(f"OUTPUT_DIR = {OUTPUT_DIR}")
    if args.force:
        backup_existing()
    ensure_split_csvs()
    if not verify():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
