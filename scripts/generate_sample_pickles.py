#!/usr/bin/env python
"""
Script to generate sample pickle files for testing the E2B Pickle Analyzer.
This creates different types of data structures in pickle format.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path


def create_sample_dir():
    """Create the sample directory if it doesn't exist."""
    sample_dir = Path("samples")
    sample_dir.mkdir(exist_ok=True)
    return sample_dir


def create_dict_pickle(sample_dir):
    """Create a sample dictionary pickle file."""
    data = {
        "name": "Sample Dictionary",
        "values": [1, 2, 3, 4, 5],
        "nested_dict": {"a": 1, "b": 2},
        "boolean": True,
        "none_value": None,
    }

    file_path = sample_dir / "dict_sample.pickle"
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

    print(f"Created dictionary pickle: {file_path}")


def create_list_pickle(sample_dir):
    """Create a sample list pickle file."""
    data = [
        "string item",
        123,
        {"nested_dict": "value"},
        [1, 2, 3],
        None,
        True,
    ]

    file_path = sample_dir / "list_sample.pickle"
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

    print(f"Created list pickle: {file_path}")


def create_numpy_pickle(sample_dir):
    """Create a sample NumPy array pickle file."""
    # Create a random 3D array
    data = np.random.rand(10, 5, 3)

    file_path = sample_dir / "numpy_sample.pickle"
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

    print(f"Created NumPy array pickle: {file_path}")


def create_pandas_pickle(sample_dir):
    """Create a sample pandas DataFrame pickle file."""
    # Create a DataFrame with different data types
    data = pd.DataFrame(
        {
            "int_col": np.random.randint(0, 100, size=20),
            "float_col": np.random.rand(20),
            "str_col": [f"item_{i}" for i in range(20)],
            "bool_col": np.random.choice([True, False], size=20),
            "cat_col": pd.Categorical(np.random.choice(["A", "B", "C"], size=20)),
            "date_col": pd.date_range(start="2023-01-01", periods=20),
        }
    )

    file_path = sample_dir / "pandas_sample.pickle"
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

    print(f"Created pandas DataFrame pickle: {file_path}")


def create_complex_pickle(sample_dir):
    """Create a complex nested data structure pickle file."""
    # Create a complex nested structure
    data = {
        "metadata": {
            "name": "Complex Dataset",
            "version": "1.0.0",
            "created_at": pd.Timestamp.now(),
        },
        "data": {
            "array_1d": np.random.rand(10),
            "array_2d": np.random.rand(5, 5),
            "dataframe": pd.DataFrame(
                {
                    "A": np.random.rand(5),
                    "B": np.random.randint(0, 10, size=5),
                }
            ),
        },
        "parameters": {
            "alpha": 0.01,
            "beta": 1.5,
            "use_normalization": True,
            "layers": [10, 5, 1],
        },
    }

    file_path = sample_dir / "complex_sample.pickle"
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

    print(f"Created complex pickle: {file_path}")


def main():
    """Generate all sample pickle files."""
    sample_dir = create_sample_dir()

    # Create different types of pickle files
    create_dict_pickle(sample_dir)
    create_list_pickle(sample_dir)
    create_numpy_pickle(sample_dir)
    create_pandas_pickle(sample_dir)
    create_complex_pickle(sample_dir)

    print("\nAll sample pickle files generated in the 'samples' directory.")
    print("You can analyze them with:")
    print("e2b-hackathon samples/*.pickle")


if __name__ == "__main__":
    main()
