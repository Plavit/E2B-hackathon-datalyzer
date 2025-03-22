import os
from typing import List

from dotenv import load_dotenv
from e2b_code_interpreter import CodeInterpreter  # type: ignore


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

    with CodeInterpreter(api_key=api_key) as code_interpreter:
        # Upload each pickle file to the sandbox
        remote_paths = []
        for file_path in pickle_files:
            if verbose:
                print(f"Uploading {file_path} to sandbox...")

            with open(file_path, "rb") as f:
                remote_path = code_interpreter.upload_file(f)
                remote_paths.append((os.path.basename(file_path), remote_path))
                if verbose:
                    print(f"  Uploaded to {remote_path}")

        # Generate and execute code to analyze the pickle files
        analysis_code = _generate_analysis_code(remote_paths)

        if verbose:
            print("Executing analysis code...")
            print("-" * 40)
            print(analysis_code)
            print("-" * 40)

        exec_result = code_interpreter.notebook.exec_cell(
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

        for result in exec_result.results:
            if hasattr(result, "data"):
                if "text/plain" in result.data:
                    print(result.data["text/plain"])
                elif "text/html" in result.data:
                    print("HTML output available (not displayed in console)")
            else:
                print(result)


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
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
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
            result['sample_keys'] = list(data.keys())[:5] if len(data) > 0 else []
        elif isinstance(data, (list, tuple)):
            result['length'] = len(data)
            result['element_types'] = list(set(type(item).__name__ for item in data[:100]))
            result['sample_elements'] = data[:5] if len(data) > 0 else []
        elif isinstance(data, pd.DataFrame):
            result['shape'] = data.shape
            result['columns'] = list(data.columns)
            result['dtypes'] = {col: str(dtype) for col, dtype in data.dtypes.items()}
            result['sample_data'] = data.head(5).to_dict() if not data.empty else {}
        elif isinstance(data, np.ndarray):
            result['shape'] = data.shape
            result['dtype'] = str(data.dtype)
            result['sample_data'] = data.flatten()[:5].tolist() if data.size > 0 else []
        
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

    # Add code to analyze each file
    for original_name, remote_path in remote_paths:
        code += f"""
print(f"Analyzing {original_name}...")
results["{original_name}"] = analyze_pickle("{remote_path}")
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
                print(pd.DataFrame(result['sample_data']))
        
        if 'dtype' in result:  # NumPy array
            print(f"Data type: {result['dtype']}")
            if result['sample_data']:
                print(f"Sample data: {result['sample_data']}")
"""

    return code
