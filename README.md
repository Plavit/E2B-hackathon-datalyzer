# D2MA Deep Data Mess Analyzer for the E2B Hackathon

A comprehensive tool for analyzing data files along with text/document context, finding correlations, and generating AI-powered analysis plans.

## Problem Statement
Were you ever thrown into a problem head-first, needing to get yourself oriented in a new space? Be it for science, for business or just for a fun side-project, that can be a start of a gruelling, weeks long process of data curation, analysis and understanding that can suck the joy out of even the most exciting of projects:

TBD image

Well WORRY NOT, for we have developed a ROBUST(🤞™️) OPEN SOURCE (📖😮) AI-FIRST Deep Data Analyzer! How does it work?
 1) Upload all data to analyze
 2) See high level analysis of relationships and data interpretation
 3) Perform agentic follow-up analysis

TBD image

Data types that are supported:
- csv
- pickle
- xml
- json
- pdf (?)
- docx (?)
- Relevant public datasets for deep search (?)

Would be nice, but better luck next time:
- External databases via API keys
- snowflake for all snowflakes out there

## How we built it
It is an AI hackathon, we used AI (cursor FTW), vibe coding and coffee. Anyway, specifically:
- agentic backend is via E2B sandboxing (obviously) and LLM API calls
- internal logic is vibe-coded in python
- frontend is using Plotly dash

Start of hackathon when we refined the idea: ca. 🕚11AM:
TBD image

End of hackathon, ca. 🕠17\:30:01 (yes, 17:30 was submission deadline how did you guess?):
TBD image

## Features

- **Multi-format Data Analysis**: Supports pickles, CSV, and Parquet files
- **Context Integration**: Import text, PDF, and Word documents to provide context for your data
- **Correlation Detection**: Automatically identifies relationships between data files
- **AI-powered Analysis Plans**: Uses OpenAI to generate analysis plans based on your data and context
- **E2B Sandbox Execution**: Safely executes generated code in an isolated sandbox environment

## Installation

1. Clone this repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
4. Install dependencies with UV: `uv add pandas pyarrow fastparquet python-docx PyPDF2 openai dash dash-bootstrap-components`
5. Set up your `.env` file with API keys

## API Keys

You'll need the following API keys:
- **E2B API Key**: For the sandbox environment
- **OpenAI API Key**: For generating analysis plans

Create a `.env` file in the project root with:
```
E2B_API_KEY=your_e2b_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

1. Start the application: `e2b-hackathon web`
2. Open your browser at `http://127.0.0.1:8050/`
3. Upload data files (pickle, CSV, Parquet) in the data section
4. Upload context files (text, PDF, Word) in the context section
5. Review the initial analysis and correlations
6. Generate an AI analysis plan
7. Execute the plan in the E2B sandbox to get results

## Workflow

1. **Data Analysis**: Upload and analyze your data files
2. **Context Integration**: Add text and document files for context
3. **Correlation Analysis**: Discover relationships between your data files
4. **AI Planning**: Generate an analysis plan based on data and context
5. **E2B Execution**: Run the generated code in a secure sandbox

## Security

All analysis of pickle files is performed securely in an E2B sandbox to prevent malicious code execution. The application never directly unpickles user-uploaded files on the server.
