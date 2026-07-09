"""
collect_checkpoints.py

Recursively finds all `checkpoints/` directories under a given parent path
and dumps all their files flat into a single output directory.

Usage:
    python collect_checkpoints.py --parent-dir /path/to/parent
    python collect_checkpoints.py --parent-dir /path/to/parent --copy-to /path/to/output
"""

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Collect all files from checkpoints/ dirs into a single flat folder."
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

    print(f"Found {len(checkpoints)} checkpoint dir(s). Dumping files to: {dest}\n")
    dest.mkdir(parents=True, exist_ok=True)

    total = 0
    for i, cp in enumerate(checkpoints, 1):
        files = [f for f in cp.iterdir() if f.is_file()]
        print(f"  [{i}/{len(checkpoints)}] {cp.relative_to(parent)}  ({len(files)} files)")
        for f in files:
            target = dest / f.name
            # Avoid collisions: prefix with parent dirs if name already exists
            if target.exists():
                label  = str(cp.relative_to(parent)).replace("/", "_")
                target = dest / f"{label}_{f.name}"
            shutil.copy2(f, target)
            total += 1

    print(f"\nDone. {total} file(s) copied to:\n  {dest}")


if __name__ == "__main__":
    main()