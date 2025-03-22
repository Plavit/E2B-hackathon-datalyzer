# E2B Deep Data Project Mess Analyzer

A secure tool for analyzing pickle files using E2B sandbox. This project allows you to analyze pickle files safely in an isolated environment, without risking exposure to potentially malicious code.

## Problem Statement
Were you ever thrown into a problem head-first, needing to ? Be it for science, for business or just for a fun side-project, that can be a start of a gruelling, weeks long process of data curation, analysis and understanding that can suck the joy out of even the most exciting of projects:

TBD image

Well WORRY NOT, for we have developed a ROBUST(🤞™️) OPEN SOURCE (📖😮) AI-FIRST Deep Data Analyzer! How does it work?
1

TBD image

## How we built it
It is an AI hackathon, we used AI (cursor FTW), vibe coding and coffee. Anyway, specifically:
- agentic backend is via E2B sandboxing (obviously) and LLM API calls
- internal logic is vibe-coded in python
- frontend is using Plotly dash

Start of hackathon when we refined the idea: ca. 11AM:
TBD image

End of hackathon, ca. 🕠17\:30:01 (yes, 17:30 was deadline):
TBD image

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
