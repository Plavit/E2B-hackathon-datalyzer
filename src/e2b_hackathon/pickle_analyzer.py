import os
from typing import List, Dict, Any
import structlog
import pickle
import sys

from dotenv import load_dotenv
from e2b_code_interpreter import Sandbox  # type: ignore


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
        raise ValueError(
            "E2B_API_KEY not set in environment variables. Please add it to your .env file."
        )

    if verbose:
        print("Initializing E2B sandbox...")

    code_interpreter = Sandbox()

    # Upload each pickle file to the sandbox
    remote_paths = []
    for file_path in pickle_files:
        if verbose:
            print(f"Uploading {file_path} to sandbox...")

        # Verify the file locally first
        try:
            with open(file_path, "rb") as f:
                file_content = f.read()
                file_size = len(file_content)

            if verbose:
                print(f"Local file size: {file_size} bytes")
                print(
                    f"Local file header (first 20 bytes): {' '.join(f'{b:02x}' for b in file_content[:20])}"
                )

                # Try to load locally to verify it's a valid pickle
                try:
                    with open(file_path, "rb") as f:
                        local_data = pickle.load(f)
                    print(
                        f"Successfully loaded pickle locally: {type(local_data).__name__}"
                    )
                except Exception as e:
                    print(f"Error loading pickle locally: {str(e)}")
        except Exception as e:
            print(f"Error reading local file {file_path}: {str(e)}")
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
            if verbose:
                print("Executing code to create file:")
                print(code)

            result = code_interpreter.run_code(code)

            if verbose:
                print(
                    f"Result: {result.text if hasattr(result, 'text') else 'No output'}"
                )

            remote_paths.append((file_name, remote_path))

            if verbose:
                print(f"  Uploaded to {remote_path}")

        except Exception as e:
            print(f"Error uploading {file_path}: {str(e)}")
            import traceback

            traceback.print_exc()
            continue

    if not remote_paths:
        print("No files were successfully uploaded. Exiting.")
        return

    # Generate and execute code to analyze the pickle files
    analysis_code = _generate_analysis_code(remote_paths)

    if verbose:
        print("Executing analysis code...")
        print("-" * 40)
        print(analysis_code)
        print("-" * 40)

    # Use run_code instead of notebook.exec_cell
    exec_result = code_interpreter.run_code(
        analysis_code,
        on_stdout=lambda stdout: print(stdout) if verbose else None,
        on_stderr=lambda stderr: print(f"Error: {stderr}") if stderr else None,
    )

    if exec_result.error:
        print(f"Error analyzing pickle files: {exec_result.error}")
        return

    # Process and display results
    if verbose:
        print("\nAnalysis results:")

    # Output the result text
    if hasattr(exec_result, "text"):
        print(exec_result.text)
    elif hasattr(exec_result, "results"):
        for result in exec_result.results:
            if hasattr(result, "data"):
                if "text/plain" in result.data:
                    print(result.data["text/plain"])
                elif "text/html" in result.data:
                    print("HTML output available (not displayed in console)")
            else:
                print(result)
    else:
        print("No results were returned.")


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
from collections import defaultdict
from typing import Any, Dict, List

def analyze_pickle(file_path: str) -> Dict[str, Any]:
    '''Analyze a pickle file and return its properties.'''
    try:
        print(f"Opening file: {file_path} (size: {os.path.getsize(file_path)} bytes)")
        
        # Debug: Read and print file header
        with open(file_path, 'rb') as f:
            header_bytes = f.read(30)
            print(f"File header (first 30 bytes): {' '.join(f'{b:02x}' for b in header_bytes)}")
            f.seek(0)
            
            # Try different pickle protocols
            for protocol in range(5):
                try:
                    print(f"Attempting pickle protocol {protocol}...")
                    f.seek(0)
                    data = pickle.load(f)
                    print(f"Successfully loaded with protocol {protocol}")
                    break
                except Exception as e:
                    print(f"Protocol {protocol} failed: {str(e)}")
                    if protocol == 4:  # Last protocol attempt
                        print("All protocols failed, returning error")
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
print(f"Analyzing {original_name}...")
results["{original_name}"] = analyze_pickle({repr(remote_path)})
"""

    # Add code to print results
    code += """
# Print results in a readable format
for file_name, result in results.items():
    print(f"\\n{'=' * 50}")
    print(f"Analysis of {file_name}:")
    print(f"{'=' * 50}")
    
    if 'error' in result:
        print(f"Error: {result['error']}")
        continue
    
    print(f"Type: {result['type']}")
    print(f"Size: {result['size_bytes']} bytes")
    
    if 'keys_count' in result:
        print(f"Number of keys: {result['keys_count']}")
        print(f"Key types: {', '.join(result['key_types'])}")
        print(f"Value types: {', '.join(result['value_types'])}")
        if result['sample_keys']:
            print(f"Sample keys: {result['sample_keys']}")
    
    if 'length' in result:
        print(f"Length: {result['length']}")
        print(f"Element types: {', '.join(result['element_types'])}")
        if result['sample_elements']:
            print(f"Sample elements: {result['sample_elements']}")
    
    if 'shape' in result:
        print(f"Shape: {result['shape']}")
        
        if 'columns' in result:  # DataFrame
            print(f"Columns: {result['columns']}")
            print("Data types:")
            for col, dtype in result['dtypes'].items():
                print(f"  - {col}: {dtype}")
            if result['sample_data']:
                print("Sample data (first 5 rows):")
                print(result['sample_data'])
        
        if 'dtype' in result:  # NumPy array
            print(f"Data type: {result['dtype']}")
            if result['sample_data']:
                print(f"Sample data: {result['sample_data']}")
"""

    return code
