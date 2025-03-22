# E2B Data Project Mess Analyzer

A secure tool for analyzing pickle files using E2B sandbox. This project allows you to analyze pickle files safely in an isolated environment, without risking exposure to potentially malicious code.

## Problem Statement
TBD

## How we built it
It is an AI hackathon, we used AI (cursor FTW), vibe coding and coffee.
TBD

## Features

- **Secure Analysis**: Uses E2B sandbox to safely analyze pickle files
- **Web Interface**: Interactive Dash web UI for uploading and analyzing files
- **Command Line Interface**: CLI for batch analysis of multiple files
- **Detailed Information**: Provides comprehensive details about pickle file contents
- **Support for Various Types**: Analyzes dictionaries, lists, DataFrames, NumPy arrays, and more

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/e2b-hackathon.git
cd e2b-hackathon

# Set up your environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv sync
```

## API Key Setup

You'll need an E2B API key to use this tool. Create a `.env` file in the root directory with the following content:

```
E2B_API_KEY=your_api_key_here
```

## Usage

### Web Interface

```bash
# Launch the web interface
e2b-hackathon web

# Advanced options
e2b-hackathon web --debug --host 0.0.0.0 --port 8080
```

Visit http://127.0.0.1:8050/ in your browser to access the web interface.

### Command Line Interface

```bash
# Analyze pickle files via CLI
e2b-hackathon analyze path/to/file1.pkl path/to/file2.pkl

# Enable verbose output
e2b-hackathon analyze -v path/to/file.pkl
```

### Sample Pickle Files

For testing purposes, you can generate sample pickle files with different data structures:

```bash
# Generate sample pickle files
./scripts/create_sample_pickle.py

# Analyze the sample files
e2b-hackathon analyze sample_pickles/*.pkl

# Or upload them through the web interface
e2b-hackathon web
```

## Security

This tool analyzes pickle files in an isolated E2B sandbox environment, providing protection against potentially malicious pickle files that could execute arbitrary code when unpickled.
