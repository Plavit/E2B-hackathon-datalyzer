#!/usr/bin/env python3
import os
from e2b_code_interpreter import Sandbox

# Load API key from environment
api_key = os.environ.get("E2B_API_KEY")
if not api_key:
    raise ValueError("E2B_API_KEY not set in environment")

print("Creating sandbox...")
sandbox = Sandbox()

print("Uploading a test file...")
with open("test_e2b.py", "rb") as f:
    file_content = f.read()
    # Try different approaches to upload the file
    try:
        print("Method 1: Using file content")
        path1 = sandbox.files.write(content=file_content, path="test1.py")
        print(f"Success! Uploaded to {path1}")
    except Exception as e:
        print(f"Method 1 failed: {e}")

    try:
        print("\nMethod 2: Using file object")
        f.seek(0)  # Reset file position
        path2 = sandbox.files.write(file=f, path="test2.py")
        print(f"Success! Uploaded to {path2}")
    except Exception as e:
        print(f"Method 2 failed: {e}")

    try:
        print("\nMethod 3: Using filename")
        path3 = sandbox.files.write("test_e2b.py")
        print(f"Success! Uploaded to {path3}")
    except Exception as e:
        print(f"Method 3 failed: {e}")

print("\nShutting down sandbox...")
sandbox.close()
print("Done!")
