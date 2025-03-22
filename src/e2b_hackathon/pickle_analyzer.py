import os
import json
from typing import List, Dict, Any
import structlog
import pickle
import sys
import traceback

from dotenv import load_dotenv
from e2b_code_interpreter import Sandbox  # type: ignore

# Configure structlog
# Set up structlog to output JSON for consistency
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
)

log = structlog.get_logger()


def analyze_pickle_files(pickle_files: List[str], verbose: bool = False) -> None:
    """
    Upload pickle files to E2B sandbox and analyze their properties.

    Args:
        pickle_files: List of paths to pickle files to analyze
        verbose: Whether to display verbose output
    """
    # Load API key from environment variables
    load_dotenv()

    # Initialize the code interpreter
    api_key = os.environ.get("E2B_API_KEY")
    if not api_key:
        log.error("E2B_API_KEY not set in environment variables")
        raise ValueError(
            "E2B_API_KEY not set in environment variables. Please add it to your .env file."
        )

    log.debug("Initializing E2B sandbox")
    code_interpreter = Sandbox()

    # Upload each pickle file to the sandbox
    remote_paths = []
    for file_path in pickle_files:
        log.debug("Processing pickle file", file_path=file_path)

        # Verify the file locally first
        try:
            with open(file_path, "rb") as f:
                file_content = f.read()
                file_size = len(file_content)

            log.debug(
                "Local file info",
                file_size=file_size,
                file_header_hex=" ".join(f"{b:02x}" for b in file_content[:20]),
            )

            # Try to load locally to verify it's a valid pickle
            try:
                with open(file_path, "rb") as f:
                    local_data = pickle.load(f)
                log.debug(
                    "Successfully loaded pickle locally", type=type(local_data).__name__
                )
            except Exception as e:
                log.warning("Error loading pickle locally", error=str(e))
        except Exception as e:
            log.error("Error reading local file", file_path=file_path, error=str(e))
            continue

        # Convert Path object to string if needed
        file_path_str = str(file_path)
        file_name = os.path.basename(file_path_str)
        remote_path = f"/tmp/{file_name}"

        try:
            # Create a temporary file in the sandbox with the pickle content
            code = f"""
import os

# Ensure directory exists
os.makedirs('/tmp', exist_ok=True)

# Write pickle data to file
with open('{remote_path}', 'wb') as f:
    f.write({repr(file_content)})
    
# Verify file was written
file_size = os.path.getsize('{remote_path}')
print(f"File written to {remote_path} ({file_size} bytes)")
"""
            log.debug(
                "Executing code to create file in sandbox", remote_path=remote_path
            )

            result = code_interpreter.run_code(code)

            if hasattr(result, "text"):
                log.debug("Sandbox file creation result", result=result.text)

            remote_paths.append((file_name, remote_path))
            log.debug(
                "File uploaded to sandbox", file_name=file_name, remote_path=remote_path
            )

        except Exception as e:
            log.error(
                "Error uploading file to sandbox",
                file_path=file_path,
                error=str(e),
                traceback=traceback.format_exc(),
            )
            continue

    if not remote_paths:
        log.error("No files were successfully uploaded")
        return

    # Generate and execute code to analyze the pickle files
    analysis_code = _generate_analysis_code(remote_paths)
    log.debug("Generated analysis code", code_length=len(analysis_code))

    def stdout_handler(stdout: Any) -> None:
        """Handle stdout from sandbox execution"""
        # Convert to string if it's not already a string
        stdout_text = str(stdout) if not isinstance(stdout, str) else stdout

        try:
            # Only print structured output, log debugging info
            if stdout_text.strip().startswith("="):
                # This is our structured separator, output it directly
                print(stdout_text.rstrip())
            elif any(
                key in stdout_text
                for key in [
                    "Object Type:",
                    "Size (bytes):",
                    "Length:",
                    "Shape:",
                    "Number of Keys:",
                ]
            ):
                # This is part of our structured output format
                print(stdout_text.rstrip())
            else:
                # This is debug information
                log.debug("Sandbox execution output", output=stdout_text.rstrip())
        except Exception as e:
            # If we have any issues processing the output, log it and continue
            log.error(
                "Error processing sandbox output",
                error=str(e),
                output_type=type(stdout).__name__,
            )
            # Try to output it anyway in case it's important
            try:
                print(stdout_text)
            except:
                log.error("Could not print stdout", output_repr=repr(stdout))

    def stderr_handler(stderr: Any) -> None:
        """Handle stderr from sandbox execution"""
        if stderr:
            try:
                stderr_text = str(stderr) if not isinstance(stderr, str) else stderr
                log.error("Sandbox execution error", error=stderr_text.rstrip())
            except Exception as e:
                log.error(
                    "Error processing stderr",
                    error=str(e),
                    stderr_type=type(stderr).__name__,
                )

    # Use run_code to execute the analysis
    exec_result = code_interpreter.run_code(
        analysis_code,
        on_stdout=stdout_handler,
        on_stderr=stderr_handler,
    )

    if exec_result.error:
        log.error("Error analyzing pickle files", error=exec_result.error)
        return

    log.debug("Analysis completed successfully")


def _generate_analysis_code(remote_paths: List[tuple[str, str]]) -> str:
    """
    Generate Python code to analyze pickle files in the sandbox.

    Args:
        remote_paths: List of (original_filename, remote_path) tuples

    Returns:
        Python code string to execute in the sandbox
    """
    # Build import statements
    code = """
import pickle
import os
import sys
import numpy as np
import pandas as pd
import json
from collections import defaultdict
from typing import Any, Dict, List

def analyze_pickle(file_path: str) -> Dict[str, Any]:
    '''Analyze a pickle file and return its properties.'''
    try:
        # Log debug info to stderr to keep stdout clean
        sys.stderr.write(f"Opening file: {file_path} (size: {os.path.getsize(file_path)} bytes)\\n")
        
        # Read file header
        with open(file_path, 'rb') as f:
            header_bytes = f.read(30)
            sys.stderr.write(f"File header: {' '.join(f'{b:02x}' for b in header_bytes)}\\n")
            f.seek(0)
            
            # Try different pickle protocols
            for protocol in range(5):
                try:
                    sys.stderr.write(f"Attempting pickle protocol {protocol}...\\n")
                    f.seek(0)
                    data = pickle.load(f)
                    sys.stderr.write(f"Successfully loaded with protocol {protocol}\\n")
                    break
                except Exception as e:
                    sys.stderr.write(f"Protocol {protocol} failed: {str(e)}\\n")
                    if protocol == 4:  # Last protocol attempt
                        sys.stderr.write("All protocols failed, returning error\\n")
                        return {
                            'file_path': file_path,
                            'error': f"Pickle load error: {str(e)}",
                            'file_size': os.path.getsize(file_path),
                            'traceback': str(sys.exc_info()[2])
                        }
        
        # If we got here, one of the protocols worked
        result = {
            'file_path': file_path,
            'type': type(data).__name__,
            'size_bytes': os.path.getsize(file_path),
        }
        
        # Add type-specific analysis
        if isinstance(data, dict):
            result['keys_count'] = len(data)
            result['key_types'] = list(set(type(k).__name__ for k in data.keys()))
            result['value_types'] = list(set(type(v).__name__ for v in data.values()))
            result['sample_keys'] = [str(k)[:100] for k in list(data.keys())[:5]] if len(data) > 0 else []
        elif isinstance(data, (list, tuple)):
            result['length'] = len(data)
            result['element_types'] = list(set(type(item).__name__ for item in data[:100]))
            # Convert elements to strings to avoid serialization issues with complex objects
            result['sample_elements'] = [str(e)[:100] for e in data[:5]] if len(data) > 0 else []
        elif isinstance(data, pd.DataFrame):
            result['shape'] = data.shape
            result['columns'] = list(data.columns)
            result['dtypes'] = {str(col): str(dtype) for col, dtype in data.dtypes.items()}
            # Convert DataFrames to_dict with orient='records' for better serialization
            try:
                result['sample_data'] = data.head(5).to_dict(orient='records') if not data.empty else {}
            except:
                result['sample_data'] = str(data.head(5))
        elif isinstance(data, np.ndarray):
            result['shape'] = data.shape
            result['dtype'] = str(data.dtype)
            # Convert numpy arrays to lists for better serialization
            try:
                flat_data = data.flatten()[:5]
                result['sample_data'] = [float(x) if np.isscalar(x) else str(x) for x in flat_data]
            except:
                result['sample_data'] = str(data.flatten()[:5])
        
        return result
    except Exception as e:
        return {
            'file_path': file_path,
            'error': str(e),
            'traceback': str(sys.exc_info()[2])
        }

# Analyze each pickle file
results = {}
"""

    # Add code to analyze each file - using raw strings for paths
    for original_name, remote_path in remote_paths:
        code += f"""
sys.stderr.write(f"Analyzing {original_name}...\\n")
results["{original_name}"] = analyze_pickle({repr(remote_path)})
"""

    # Add code to print results in a consistent, parsable format
    code += """
# Print results in a structured, parsable format
for file_name, result in results.items():
    print(f"\\n{'=' * 50}")
    print(f"Analysis of {file_name}:")
    print(f"{'=' * 50}")
    
    if 'error' in result:
        print(f"Error: {result['error']}")
        continue
    
    print(f"Object Type: {result['type']}")
    print(f"Size (bytes): {result['size_bytes']}")
    
    if 'keys_count' in result:
        print(f"Number of Keys: {result['keys_count']}")
        print(f"Key Types: {', '.join(result['key_types'])}")
        print(f"Value Types: {', '.join(result['value_types'])}")
        print(f"Sample Keys: {result['sample_keys']}")
    
    if 'length' in result:
        print(f"Length: {result['length']}")
        print(f"Element Types: {', '.join(result['element_types'])}")
        print(f"Sample Elements: {result['sample_elements']}")
    
    if 'shape' in result:
        print(f"Shape: {result['shape']}")
        
        if 'columns' in result:  # DataFrame
            print(f"Columns: {result['columns']}")
            print(f"Data Types:")
            for col, dtype in result['dtypes'].items():
                print(f"  - {col}: {dtype}")
            if 'sample_data' in result:
                print(f"Sample Data: {result['sample_data']}")
        
        if 'dtype' in result:  # NumPy array
            print(f"Data Type: {result['dtype']}")
            if 'sample_data' in result:
                print(f"Sample Data: {result['sample_data']}")
"""

    return code
