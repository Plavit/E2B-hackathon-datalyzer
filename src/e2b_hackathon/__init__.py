import argparse
import os
import sys
from pathlib import Path

from .pickle_analyzer import analyze_pickle_files


def main() -> None:
    """Entry point for the E2B pickle analyzer command-line interface."""
    parser = argparse.ArgumentParser(
        description="Analyze pickle files using E2B sandbox."
    )
    parser.add_argument(
        "pickle_files",
        nargs="+",
        type=Path,
        help="Path(s) to pickle file(s) to analyze",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    # Validate that all files exist
    invalid_files = []
    for file_path in args.pickle_files:
        if not os.path.exists(file_path):
            invalid_files.append(file_path)

    if invalid_files:
        print("Error: The following files do not exist:", file=sys.stderr)
        for file_path in invalid_files:
            print(f"  - {file_path}", file=sys.stderr)
        sys.exit(1)

    analyze_pickle_files(pickle_files=args.pickle_files, verbose=args.verbose)
