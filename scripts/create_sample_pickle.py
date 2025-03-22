#!/usr/bin/env python3
import os
import pickle
import numpy as np
import pandas as pd

"""
This script creates sample pickle files for testing the E2B Pickle Analyzer.
It generates different types of data structures and saves them as pickle files.
"""


def create_sample_dict():
    """Create a sample dictionary with various data types."""
    sample_dict = {
        "string_key": "This is a string value",
        "int_key": 42,
        "float_key": 3.14159,
        "bool_key": True,
        "list_key": [1, 2, 3, 4, 5],
        "dict_key": {"nested": "value"},
        "none_key": None,
    }
    return sample_dict


def create_sample_list():
    """Create a sample list with various data types."""
    sample_list = ["string value", 42, 3.14159, True, [1, 2, 3], {"key": "value"}, None]
    return sample_list


def create_sample_dataframe():
    """Create a sample pandas DataFrame."""
    data = {
        "Name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "Age": [25, 30, 35, 40, 45],
        "City": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"],
        "Salary": [50000.0, 60000.0, 70000.0, 80000.0, 90000.0],
        "Employed": [True, True, False, True, False],
    }
    return pd.DataFrame(data)


def create_sample_numpy_array():
    """Create a sample numpy array."""
    return np.random.rand(5, 3)


def main():
    # Create output directory if it doesn't exist
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "sample_pickles"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Generate and save sample pickle files
    samples = {
        "sample_dict.pkl": create_sample_dict(),
        "sample_list.pkl": create_sample_list(),
        "sample_dataframe.pkl": create_sample_dataframe(),
        "sample_numpy_array.pkl": create_sample_numpy_array(),
    }

    for filename, data in samples.items():
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"Created {filepath}")

    print("\nSample pickle files created. You can now run:")
    print("e2b-hackathon analyze sample_pickles/*.pkl")
    print("or")
    print("e2b-hackathon web")
    print("and upload the files through the web interface.")


if __name__ == "__main__":
    main()
