"""
collect_checkpoints.py

Recursively finds all `checkpoints/` directories under a given parent path
and copies them into an output directory, flat into the destination directory.

Usage:
    python collect_checkpoints.py --parent-dir /path/to/parent
    python collect_checkpoints.py --parent-dir /path/to/parent --copy-to /path/to/output
"""

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Copy all checkpoints/ dirs from a parent analysis folder."
    )
    parser.add_argument(
        "--parent-dir",
        type=Path,
        required=True,
        help="Parent directory to search for checkpoints/ folders.",
    )
    parser.add_argument(
        "--copy-to",
        type=Path,
        default=None,
        help="Destination directory (default: <parent-dir>/collected_checkpoints).",
    )
    args = parser.parse_args()

    parent: Path = args.parent_dir.resolve()
    dest: Path   = (args.copy_to or parent / "collected_checkpoints").resolve()

    if not parent.exists():
        raise FileNotFoundError(f"Source directory not found: {parent}")

    print(f"Searching: {parent}")
    checkpoints = sorted(parent.rglob("checkpoints"))

    if not checkpoints:
        print("No checkpoints/ directories found.")
        return

    print(f"Found {len(checkpoints)} checkpoint dir(s). Copying to: {dest}\n")
    dest.mkdir(parents=True, exist_ok=True)

    for i, cp in enumerate(checkpoints, 1):
        rel    = cp.relative_to(parent)
        label  = str(rel).replace("/", "_")   # flat: 241030_M07420_Network_000022_well001_checkpoints
        target = dest
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(cp, target)
        print(f"  [{i}/{len(checkpoints)}] {rel}  →  {label}")

    print(f"\nDone. {len(checkpoints)} checkpoint dir(s) copied to:\n  {dest}")


if __name__ == "__main__":
    main()