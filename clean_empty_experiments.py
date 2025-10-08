#!/usr/bin/env python3
import argparse
import os
import shutil
from pathlib import Path


def has_mp4_in_video_debug(exp_dir: Path) -> bool:
    video_dir = exp_dir / "video_debug"
    if not video_dir.is_dir():
        return False
    # Check for any .mp4 files (non-recursive)
    for p in video_dir.iterdir():
        if p.is_file() and p.suffix.lower() == ".mp4":
            return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Delete wandb sub-experiments that have no .mp4 in video_debug."
    )
    parser.add_argument(
        "wandb_root",
        nargs="?",
        default="/data1/lcy/projects/ReCamMaster/wandb",
        help="Path to wandb root directory (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be deleted, do not delete.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output."
    )
    args = parser.parse_args()

    root = Path(args.wandb_root)
    if not root.is_dir():
        print(f"Error: {root} is not a directory.")
        raise SystemExit(1)

    to_delete = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        # Treat every immediate subdirectory as a sub-experiment
        if args.verbose:
            print(f"Checking: {entry}")
        if not has_mp4_in_video_debug(entry):
            to_delete.append(entry)

    if not to_delete:
        print("No empty experiments found (no candidates for deletion).")
        return

    print("Sub-experiments to delete (no .mp4 in video_debug):")
    for d in to_delete:
        print(f" - {d}")

    if args.dry_run:
        print("\nDry run enabled: no directories were deleted.")
        return

    # Proceed to delete
    failures = 0
    for d in to_delete:
        try:
            shutil.rmtree(d)
            if args.verbose:
                print(f"Deleted: {d}")
        except Exception as e:
            failures += 1
            print(f"Failed to delete {d}: {e}")

    print(f"\nDone. Deleted {len(to_delete) - failures} directories. Failures: {failures}.")


if __name__ == "__main__":
    main()

