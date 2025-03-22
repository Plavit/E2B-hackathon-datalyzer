import argparse
import os
import sys
from pathlib import Path

from e2b_hackathon.pickle_analyzer import analyze_pickle_files


def main() -> None:
    """Entry point for the E2B pickle analyzer command-line interface."""
    parser = argparse.ArgumentParser(
        description="Analyze pickle files using E2B sandbox."
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # CLI analyzer command
    cli_parser = subparsers.add_parser("analyze", help="Analyze pickle files via CLI")
    cli_parser.add_argument(
        "pickle_files",
        nargs="+",
        type=Path,
        help="Path(s) to pickle file(s) to analyze",
    )
    cli_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    # Web UI command
    web_parser = subparsers.add_parser(
        "web", help="Launch web UI for analyzing pickle files"
    )
    web_parser.add_argument(
        "--debug", action="store_true", help="Run the web server in debug mode"
    )
    web_parser.add_argument(
        "--host", default="127.0.0.1", help="Host for the web server"
    )
    web_parser.add_argument(
        "--port", type=int, default=8050, help="Port for the web server"
    )

    args = parser.parse_args()

    # Default to 'analyze' if no command provided
    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "analyze":
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

    elif args.command == "web":
        from e2b_hackathon.dash_app import app

        app.run(debug=args.debug, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
