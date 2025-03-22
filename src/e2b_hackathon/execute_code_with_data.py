"""
Module for executing code in E2B sandbox with data files.

This module provides functions to securely execute Python code in an E2B sandbox
while uploading and providing access to local data files.
"""

import io
import os
import base64
import json
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import traceback
import structlog

# Configure structlog
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
)
logger = structlog.get_logger()


def execute_code_with_data(
    code: str, data_paths: Dict[str, str], capture_plots: bool = True
) -> Dict[str, Any]:
    """
    Execute Python code in an E2B sandbox with access to local data files.

    Args:
        code: Python code to execute
        data_paths: Dictionary mapping file names to local file paths
        capture_plots: Whether to capture matplotlib plots

    Returns:
        Dictionary with execution results containing:
        - status: "success" or "error"
        - timestamp: ISO format timestamp
        - stdout: Standard output as string
        - stderr: Standard error as string
        - error: Error message if any
        - plots: List of base64-encoded plot images if capture_plots=True
        - tables: List of table data if any tables were generated
    """
    try:
        # Type ignore for missing stubs in e2b_code_interpreter
        from e2b_code_interpreter import Sandbox  # type: ignore
    except ImportError:
        error_msg = "Missing dependency: e2b_code_interpreter. Install with: pip install e2b-code-interpreter"
        logger.error(error_msg)
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": error_msg,
        }

    logger.info("Initializing E2B sandbox for code execution")

    # Create a sandbox instance
    sandbox = Sandbox()

    # Prepare results container
    results: Dict[str, Any] = {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "stdout": "",
        "stderr": "",
        "error": None,
        "plots": [],
        "tables": [],
    }

    try:
        # Upload data files to the sandbox if provided
        remote_paths = {}
        if data_paths:
            logger.info(f"Uploading {len(data_paths)} data files to sandbox")

            for file_name, file_path in data_paths.items():
                try:
                    # Read the file content
                    with open(file_path, "rb") as f:
                        file_content = f.read()

                    # Create a remote path
                    remote_path = f"/tmp/{file_name}"

                    # Write file to the sandbox
                    file_upload_code = f"""
import os

# Ensure tmp directory exists
os.makedirs('/tmp', exist_ok=True)

# Write data to file
with open('{remote_path}', 'wb') as f:
    f.write({repr(file_content)})
"""
                    sandbox.run_code(file_upload_code)

                    # Store the remote path
                    remote_paths[file_name] = remote_path
                    logger.info(f"Uploaded {file_name} to sandbox at {remote_path}")

                except Exception as e:
                    logger.error(f"Error uploading {file_name}: {str(e)}")
                    results["stderr"] += f"Error uploading {file_name}: {str(e)}\n"

            # Modify the code to use the remote paths
            for file_name, remote_path in remote_paths.items():
                # Replace file paths in the code
                code = code.replace(file_path, remote_path)

        # Execute the code in the sandbox - no need for special plot capture code
        logger.info(f"Executing code in E2B sandbox (code length: {len(code)})")

        # Capture stdout and stderr
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        def on_stdout(stdout: str) -> None:
            stdout_buffer.write(stdout)
            results["stdout"] += stdout

        def on_stderr(stderr: str) -> None:
            stderr_buffer.write(stderr)
            results["stderr"] += stderr

        # Execute the code
        execution = sandbox.run_code(code, on_stdout=on_stdout, on_stderr=on_stderr)

        # Check for execution errors
        if execution.error:
            results["status"] = "error"
            results["error"] = str(execution.error)
            logger.error(f"Error executing code in sandbox: {str(execution.error)}")

        # E2B directly captures plots and other outputs in execution.results
        if hasattr(execution, "results"):
            # Extract all plots and tables from results
            plots = []
            tables = []

            for result in execution.results:
                if hasattr(result, "png") and result.png:
                    plots.append(result.png)
                    logger.info("Found a plot in execution results")
                elif hasattr(result, "table") and result.table:
                    tables.append(result.table)
                    logger.info("Found a table in execution results")

            results["plots"] = plots
            results["tables"] = tables

            logger.info(
                f"Extracted {len(plots)} plots and {len(tables)} tables from execution"
            )

    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"Error in execute_code_with_data: {str(e)}\n{error_traceback}")
        results["status"] = "error"
        results["error"] = str(e)
        results["traceback"] = error_traceback

    finally:
        # Close the sandbox to free resources
        try:
            sandbox.close()
            logger.info("E2B sandbox closed")
        except Exception as e:
            logger.error(f"Error closing E2B sandbox: {str(e)}")

    return results


def extract_file_dependencies(code: str) -> List[str]:
    """
    Extract potential file dependencies from code.

    This is a simple implementation that looks for common file operations.
    For more complex cases, consider using AST parsing.

    Args:
        code: Python code to analyze

    Returns:
        List of potential file paths mentioned in the code
    """
    import re

    # Common patterns for file operations
    patterns = [
        r'open\s*\(\s*[\'"](.+?)[\'"]\s*,',
        r'pd\.read_csv\s*\(\s*[\'"](.+?)[\'"]\s*',
        r'pd\.read_parquet\s*\(\s*[\'"](.+?)[\'"]\s*',
        r'pd\.read_pickle\s*\(\s*[\'"](.+?)[\'"]\s*',
        r'pd\.read_excel\s*\(\s*[\'"](.+?)[\'"]\s*',
        r'pickle\.load\s*\(\s*open\s*\(\s*[\'"](.+?)[\'"]\s*',
        r'np\.load\s*\(\s*[\'"](.+?)[\'"]\s*',
    ]

    file_deps = []
    for pattern in patterns:
        matches = re.findall(pattern, code)
        file_deps.extend(matches)

    # Remove duplicates and filter out non-files
    return list(set([f for f in file_deps if not f.startswith("http")]))


def save_plots_to_files(results: Dict[str, Any], output_dir: str) -> List[str]:
    """
    Save base64-encoded plots to image files.

    Args:
        results: Results dictionary from execute_code_with_data
        output_dir: Directory to save plot images

    Returns:
        List of saved plot file paths
    """
    saved_files: List[str] = []
    plots = results.get("plots", [])

    if not plots:
        return saved_files

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for i, plot_base64 in enumerate(plots):
        try:
            # Decode base64 string
            img_data = base64.b64decode(plot_base64)

            # Save to file
            output_path = os.path.join(output_dir, f"plot_{i + 1}.png")
            with open(output_path, "wb") as f:
                f.write(img_data)

            saved_files.append(output_path)
            logger.info(f"Saved plot to {output_path}")
        except Exception as e:
            logger.error(f"Error saving plot {i + 1}: {str(e)}")

    return saved_files


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Execute Python code in an E2B sandbox with data files."
    )

    # Required argument: code file
    parser.add_argument(
        "code_file", help="Path to Python file containing code to execute"
    )

    # Optional arguments
    parser.add_argument(
        "--data",
        "-d",
        action="append",
        nargs=2,
        metavar=("NAME", "PATH"),
        help="Data file to upload to sandbox. Specify name and path. Can be used multiple times.",
    )

    parser.add_argument(
        "--no-plots", action="store_true", help="Disable plot capturing"
    )

    parser.add_argument(
        "--auto-detect-files",
        action="store_true",
        help="Automatically detect file dependencies in code",
    )

    parser.add_argument("--save-plots-dir", help="Directory to save captured plots")

    parser.add_argument("--output-json", help="Save execution results to JSON file")

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    # Parse arguments
    args = parser.parse_args()

    # Set up logging based on verbosity
    if args.verbose:
        # structlog doesn't have a direct equivalent to setLevel
        # Instead we'll configure it to include DEBUG level logs
        structlog.configure(
            processors=[
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.dev.ConsoleRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger("DEBUG"),
        )
        logger.debug("Debug logging enabled")

    # Read code from file
    try:
        with open(args.code_file, "r") as f:
            code = f.read()
            print(f"📄 Read {len(code)} characters from {args.code_file}")
    except Exception as e:
        print(f"❌ Error reading code file: {str(e)}")
        exit(1)

    # Set up data paths
    data_paths = {}

    # Add explicitly specified data files
    if args.data:
        for name, path in args.data:
            if os.path.exists(path):
                data_paths[name] = path
                print(f"📊 Added data file: {name} -> {path}")
            else:
                print(f"⚠️ Warning: Data file not found: {path}")

    # Auto-detect files if requested
    if args.auto_detect_files:
        print("🔍 Auto-detecting file dependencies...")
        detected_files = extract_file_dependencies(code)
        for file_path in detected_files:
            if os.path.exists(file_path):
                file_name = os.path.basename(file_path)
                data_paths[file_name] = file_path
                print(f"🔎 Detected file: {file_name} -> {file_path}")

    # If no data files were specified or detected
    if not data_paths:
        print(
            "⚠️ No data files specified or detected. Executing code without data files."
        )

    # Execute the code
    print(f"🚀 Executing code in E2B sandbox with {len(data_paths)} data files...")
    results = execute_code_with_data(
        code=code, data_paths=data_paths, capture_plots=not args.no_plots
    )

    # Print execution results
    print("\n" + "=" * 50)
    print(
        f"Execution status: {'✅ Success' if results['status'] == 'success' else '❌ Error'}"
    )
    print(f"Timestamp: {results['timestamp']}")

    if results.get("error"):
        print(f"\n❌ Error: {results['error']}")
        if "traceback" in results:
            print("\nTraceback:")
            print(results["traceback"])

    if results["stdout"]:
        print("\n📝 Standard Output:")
        print("-" * 50)
        print(results["stdout"])
        print("-" * 50)

    if results["stderr"]:
        print("\n⚠️ Standard Error:")
        print("-" * 50)
        print(results["stderr"])
        print("-" * 50)

    # Handle plots
    if results.get("plots"):
        print(f"\n📊 Generated {len(results['plots'])} plots")

        # Save plots if requested
        if args.save_plots_dir:
            saved_files = save_plots_to_files(results, args.save_plots_dir)
            print(f"💾 Saved {len(saved_files)} plots to {args.save_plots_dir}")
            for path in saved_files:
                print(f"   - {path}")

    # Save results to JSON if requested
    if args.output_json:
        try:
            with open(args.output_json, "w") as f:
                # Create a copy of results without plots (they're too large for JSON)
                json_results = results.copy()
                plot_count = len(json_results.get("plots", []))

                if plot_count > 0:
                    # Store just the count instead of the actual base64 data
                    json_results["plots"] = (
                        f"{plot_count} plots (base64 data excluded from JSON)"
                    )

                json.dump(json_results, f, indent=2)
                print(f"💾 Saved execution results to {args.output_json}")
        except Exception as e:
            print(f"❌ Error saving results to JSON: {str(e)}")

    print("\n🏁 Execution complete")
