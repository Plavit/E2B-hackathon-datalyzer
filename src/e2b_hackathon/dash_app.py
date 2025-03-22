import os
import sys
import json
import base64
import traceback
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
import io

import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc  # type: ignore

# import plotly.graph_objects as go  # type: ignore
import plotly.express as px  # type: ignore
import structlog
import pandas as pd
import numpy as np

# For handling different file types
from openai import OpenAI

from e2b_hackathon.pickle_analyzer import analyze_pickle_files

# Import the execute_code_with_data function from our new module
from e2b_hackathon.execute_code_with_data import execute_code_with_data

# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="Datalyzer",
)
log = structlog.get_logger()
SEED = 42

# Add custom CSS styles
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Full height container */
            .full-height-container {
                min-height: calc(100vh - 80px); /* Account for navbar */
                display: flex;
                flex-direction: row;
            }
            
            /* Full height columns */
            .full-height-col {
                display: flex;
                flex-direction: column;
                min-height: 100%;
            }
            
            /* Full height card */
            .full-height-card {
                flex: 1;
                display: flex;
                flex-direction: column;
                margin-bottom: 0;
            }
            
            .full-height-card .card-body {
                flex: 1;
                display: flex;
                flex-direction: column;
            }
            
            /* Fixed height container for independent scrolling */
            .scroll-container {
                flex: 1;
                overflow-y: auto;
                position: relative;
            }
            
            .analysis-output {
                background-color: white;
                border-radius: 10px;
                padding: 15px;
                margin-bottom: 15px;
                box-shadow: 0 1px 2px rgba(0,0,0,0.1);
                font-size: 0.9rem;
            }
            
            .analysis-output pre {
                background-color: #f8f9fa;
                padding: 10px;
                border-radius: 5px;
                font-size: 0.85rem;
                overflow-x: auto;
            }
            
            .btn-primary {
                background-color: #10a37f;
                border-color: #10a37f;
            }
            
            .btn-primary:hover {
                background-color: #0d8c6d;
                border-color: #0d8c6d;
            }
            
            /* ChatGPT-like scrollbar */
            .scroll-container::-webkit-scrollbar {
                width: 8px;
            }
            
            .scroll-container::-webkit-scrollbar-track {
                background: #f1f1f1;
                border-radius: 4px;
            }
            
            .scroll-container::-webkit-scrollbar-thumb {
                background: #c5c5c5;
                border-radius: 4px;
            }
            
            .scroll-container::-webkit-scrollbar-thumb:hover {
                background: #a8a8a8;
            }
            
            /* Card and alert styling */
            .card {
                border: none;
                margin-bottom: 1rem;
            }
            
            .alert {
                border: none;
                border-radius: 10px;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Define the navbar with logo
navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                dbc.Row(
                    [
                        dbc.Col(html.Img(src="/assets/D2MA-logo.svg", height="40px")),
                    ],
                    align="center",
                ),
                href="/",
            ),
        ],
        fluid=True,
    ),
    color="rgb(235, 248, 235)",  # Light pastel green
    dark=False,
    className="mb-4",
    fixed="top",  # Makes navbar sticky at top
    style={"width": "100%"}  # Ensures full width
)

# Initialize OpenAI client (will use API key from environment)
try:
    openai_client: Optional[OpenAI] = OpenAI()
except Exception as e:
    log.error("Failed to initialize OpenAI client", error=str(e))
    openai_client = None  # type: ignore

# Define application layout
layout = html.Div(
    [
        navbar,  # Add the navbar at the top
        # Add margin-top to account for fixed navbar
        html.Div(style={"margin-top": "80px"}),  # Spacer for fixed navbar
        
        # Main content area with 1:2 split
        dbc.Container(
            [
                dbc.Row(
                    [
                        # Left column (1/3 width) - ChatGPT-like analysis window
                        dbc.Col(
                            html.Div(
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                # Title and description
                                                html.H2(
                                                    "Enhanced Data Analysis Platform",
                                                    className="mb-3",
                                                    style={"fontSize": "1.5rem"}
                                                ),
                                                html.P(
                                                    "Upload data files (Pickle, CSV, Parquet) and context files (Text, PDF, Word) for comprehensive analysis",
                                                    className="lead mb-4",
                                                    style={"fontSize": "0.9rem"}
                                                ),
                                                
                                                # Analysis sections in scrollable container
                                                html.Div(
                                                    [
                                                        # Correlation Analysis
                                                        html.Div(
                                                            [
                                                                html.H5("Correlation Analysis", className="mb-3"),
                                                                html.Button(
                                                                    "Analyze Correlations",
                                                                    id="analyze-correlations-button",
                                                                    className="btn btn-primary btn-sm mb-3",
                                                                ),
                                                                dcc.Loading(
                                                                    id="loading-correlation",
                                                                    type="circle",
                                                                    children=html.Div(id="correlation-loading-output"),
                                                                ),
                                                                html.Div(
                                                                    id="correlation-output",
                                                                    className="analysis-output"
                                                                ),
                                                            ],
                                                            className="mb-4",
                                                        ),
                                                        
                                                        # AI Analysis Plan
                                                        html.Div(
                                                            [
                                                                html.H5("AI Analysis Plan", className="mb-3"),
                                                                html.Button(
                                                                    "Generate Analysis Plan",
                                                                    id="generate-plan-button",
                                                                    className="btn btn-primary btn-sm mb-3",
                                                                ),
                                                                dcc.Loading(
                                                                    id="loading-plan",
                                                                    type="circle",
                                                                    children=html.Div(id="plan-loading-output"),
                                                                ),
                                                                html.Div(
                                                                    id="ai-plan-output",
                                                                    className="analysis-output"
                                                                ),
                                                            ],
                                                            className="mb-4",
                                                        ),
                                                        
                                                        # AI Analysis Results
                                                        html.Div(
                                                            [
                                                                html.H5("AI Analysis Results", className="mb-3"),
                                                                html.Div(
                                                                    id="ai-results-output",
                                                                    className="analysis-output"
                                                                ),
                                                            ],
                                                            className="mb-4",
                                                        ),
                                                    ],
                                                    className="scroll-container",
                                                ),
                                            ]
                                        )
                                    ],
                                    className="h-100 border-0 full-height-card",
                                    style={
                                        "backgroundColor": "#f8f9fa",
                                        "borderRadius": "15px",
                                        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"
                                    }
                                ),
                                className="full-height-col"
                            ),
                            width=4,
                            className="full-height-col"
                        ),
                        
                        # Right column (2/3 width)
                        dbc.Col(
                            html.Div(
                                [
                                    # File upload section
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    dbc.Card(
                                                        [
                                                            dbc.CardBody(
                                                                [
                                                                    html.H5(
                                                                        "Unified File Upload",
                                                                        className="card-title",
                                                                    ),
                                                                    html.P(
                                                                        "Upload any files (.pkl, .pickle, .csv, .parquet, .txt, .pdf, .docx, .json, .geojson) for analysis. Files will be automatically categorized based on extension.",
                                                                        className="card-text text-muted",
                                                                    ),
                                                                    dcc.Upload(
                                                                        id="unified-upload",
                                                                        children=html.Div(
                                                                            [
                                                                                html.I(
                                                                                    className="fas fa-upload me-2"
                                                                                ),
                                                                                "Drag and Drop or ",
                                                                                html.A("Select Files"),
                                                                            ]
                                                                        ),
                                                                        style={
                                                                            "width": "100%",
                                                                            "height": "60px",
                                                                            "lineHeight": "60px",
                                                                            "borderWidth": "1px",
                                                                            "borderStyle": "dashed",
                                                                            "borderRadius": "5px",
                                                                            "textAlign": "center",
                                                                            "margin": "10px 0",
                                                                        },
                                                                        multiple=True,
                                                                    ),
                                                                    html.P(
                                                                        "Files will be automatically analyzed upon upload",
                                                                        className="text-muted small mt-2",
                                                                    ),
                                                                ]
                                                            )
                                                        ],
                                                        className="mb-4",
                                                    ),
                                                ],
                                                width=12,
                                            ),
                                        ]
                                    ),
                                    
                                    # Loading and status sections
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    dcc.Loading(
                                                        id="loading-analysis",
                                                        type="circle",
                                                        children=html.Div(id="loading-output"),
                                                    ),
                                                ],
                                                width=12,
                                            )
                                        ]
                                    ),
                                    dbc.Row([dbc.Col([html.Div(id="files-being-analyzed")], width=12)]),
                                    dbc.Row([dbc.Col([html.Div(id="error-message")], width=12)]),
                                    
                                    # Results sections
                                    dbc.Row([dbc.Col([html.Div(id="data-output", className="mt-4")], width=12)]),
                                    dbc.Row([dbc.Col([html.Div(id="context-output", className="mt-4")], width=12)]),
                                    
                                    # Store components
                                    dcc.Store(id="data-store", storage_type="memory"),
                                    dcc.Store(id="context-store", storage_type="memory"),
                                    dcc.Store(id="correlation-store", storage_type="memory"),
                                    dcc.Store(id="ai-plan-store", storage_type="memory"),
                                ],
                                className="scroll-container"
                            ),
                            width=8,
                            className="full-height-col"
                        ),
                    ],
                    className="g-0 full-height-container",  # Remove gutters between columns
                ),
            ],
            fluid=True,
            className="full-height-container"
        ),
    ]
)

# Set the layout
app.layout = layout  # type: ignore


def save_uploaded_files(content, filename, file_type="data"):
    """
    Save uploaded files to temporary directory

    Args:
        content: File content
        filename: File name
        file_type: Type of file ('data' or 'context')

    Returns:
        Path to the saved file
    """
    valid_extensions = {
        "data": [
            ".pkl",
            ".pickle",
            ".csv",
            ".parquet",
            ".xls",
            ".xlsx",
            ".json",
            ".geojson",
        ],
        "context": [".txt", ".pdf", ".docx", ".doc", ".md"],
    }

    # Check extension
    _, ext = os.path.splitext(filename)
    ext = ext.lower()
    if ext not in valid_extensions[file_type]:
        raise ValueError(f"Invalid file extension for {file_type} file: {ext}")

    # Create temporary directory if it doesn't exist
    temp_dir = os.path.join(os.getcwd(), "temp_files")
    os.makedirs(temp_dir, exist_ok=True)

    # Decode and save file
    content_type, content_string = content.split(",")
    decoded = base64.b64decode(content_string)

    # Create a clean filename
    clean_filename = "".join(
        [c if c.isalnum() or c in "._- " else "_" for c in filename]
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(temp_dir, f"{timestamp}_{clean_filename}")

    # Save the file
    with open(file_path, "wb") as f:
        f.write(decoded)

    return file_path


def update_execution_results(n_clicks, plan_data, data_store):
    """
    Callback for executing the generated code in E2B sandbox.
    Returns:
        - AI results output component
    """
    if (
        n_clicks is None
        or n_clicks == 0
        or not plan_data
        or "analysis_code" not in plan_data
    ):
        return html.Div("Execute the generated code to see results here.")

    try:
        log.info("Executing analysis code in E2B sandbox")

        # Get the code to execute
        analysis_code = plan_data["analysis_code"]
        if isinstance(analysis_code, list):
            analysis_code = "\n".join(analysis_code)

        # Get data file paths
        data_paths = {}
        if data_store:
            for filename, file_info in data_store.items():
                if "Path" in file_info:
                    data_paths[filename] = file_info["Path"]

        # Execute the code using our new function
        execution_results = execute_code_with_data(analysis_code, data_paths)

        output_components = []

        # Add execution status alert
        status_color = (
            "success" if execution_results.get("status") == "success" else "danger"
        )
        output_components.append(
            dbc.Alert(
                [
                    html.H5(
                        f"Execution {'Successful' if status_color == 'success' else 'Failed'}",
                        className="alert-heading",
                    ),
                    html.P(
                        f"Executed at: {execution_results.get('timestamp', 'Unknown')}"
                    ),
                ],
                color=status_color,
                className="mb-3",
            )
        )

        # Display any error
        if execution_results.get("error"):
            output_components.append(
                dbc.Card(
                    [
                        dbc.CardHeader(
                            html.H5("Execution Error", className="text-danger mb-0")
                        ),
                        dbc.CardBody(
                            [
                                html.Pre(
                                    execution_results.get("error", "Unknown error"),
                                    style={
                                        "backgroundColor": "#f8f9fa",
                                        "padding": "10px",
                                        "borderRadius": "5px",
                                        "color": "#dc3545",
                                    },
                                ),
                                html.Hr(),
                                html.Strong("Traceback:"),
                                html.Pre(
                                    execution_results.get(
                                        "traceback", "No traceback available"
                                    ),
                                    style={
                                        "backgroundColor": "#f8f9fa",
                                        "padding": "10px",
                                        "borderRadius": "5px",
                                        "maxHeight": "300px",
                                        "overflowY": "auto",
                                        "fontSize": "12px",
                                        "color": "#6c757d",
                                    },
                                ),
                            ]
                        ),
                    ],
                    className="mb-4",
                    color="danger",
                    outline=True,
                )
            )

        # Display plots if any
        plots = execution_results.get("plots", [])
        if plots:
            plot_items = []

            for i, plot_data in enumerate(plots):
                plot_items.append(
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(f"Plot {i + 1}"),
                                dbc.CardBody(
                                    html.Img(
                                        src=f"data:image/png;base64,{plot_data}",
                                        style={"width": "100%"},
                                    )
                                ),
                            ],
                            className="mb-3",
                        ),
                        md=6,
                    )
                )

            output_components.append(
                html.Div(
                    [
                        html.H4("Generated Plots", className="mb-3"),
                        dbc.Row(plot_items),
                    ],
                    className="mb-4",
                )
            )

        # Display tables if any
        tables = execution_results.get("tables", [])
        if tables:
            table_items = []

            for i, table_data in enumerate(tables):
                try:
                    # Convert table data to HTML table
                    df = pd.DataFrame(table_data)
                    html_table = dbc.Table.from_dataframe(
                        df, striped=True, bordered=True, hover=True, responsive=True
                    )

                    table_items.append(
                        dbc.Card(
                            [
                                dbc.CardHeader(f"Table {i + 1}"),
                                dbc.CardBody(html_table),
                            ],
                            className="mb-3",
                        )
                    )
                except Exception as e:
                    log.error(f"Error converting table data to HTML: {str(e)}")

            if table_items:
                output_components.append(
                    html.Div(
                        [
                            html.H4("Generated Tables", className="mb-3"),
                            html.Div(table_items),
                        ],
                        className="mb-4",
                    )
                )

        # Display stdout
        if execution_results.get("stdout"):
            output_components.append(
                dbc.Card(
                    [
                        dbc.CardHeader(html.H5("Execution Output", className="mb-0")),
                        dbc.CardBody(
                            html.Pre(
                                execution_results.get("stdout", "No output"),
                                style={
                                    "backgroundColor": "#f8f9fa",
                                    "padding": "10px",
                                    "borderRadius": "5px",
                                    "maxHeight": "400px",
                                    "overflowY": "auto",
                                },
                            )
                        ),
                    ],
                    className="mb-4",
                )
            )

        # Display stderr if any
        if execution_results.get("stderr"):
            output_components.append(
                dbc.Card(
                    [
                        dbc.CardHeader(html.H5("Standard Error", className="mb-0")),
                        dbc.CardBody(
                            html.Pre(
                                execution_results.get("stderr", "No errors"),
                                style={
                                    "backgroundColor": "#f8f9fa",
                                    "padding": "10px",
                                    "borderRadius": "5px",
                                    "maxHeight": "300px",
                                    "overflowY": "auto",
                                    "color": "#dc3545",
                                },
                            )
                        ),
                    ],
                    className="mb-4",
                    color="warning",
                    outline=True,
                )
            )

        # Display the executed code
        output_components.append(
            dbc.Card(
                [
                    dbc.CardHeader(html.H5("Executed Code", className="mb-0")),
                    dbc.CardBody(
                        html.Pre(
                            analysis_code,
                            style={
                                "backgroundColor": "#f8f9fa",
                                "padding": "10px",
                                "borderRadius": "5px",
                                "maxHeight": "300px",
                                "overflowY": "auto",
                            },
                        )
                    ),
                ],
                className="mb-4",
            )
        )

        return html.Div(output_components)

    except Exception as e:
        log.error("Error executing code in sandbox", error=str(e))
        return dbc.Alert(
            [
                html.H5("Error Executing Code"),
                html.P(str(e)),
                html.Pre(traceback.format_exc()),
            ],
            color="danger",
        )


def analyze_data_file(file_path: str) -> Dict[str, Any]:
    """
    Analyze different types of data files.

    Args:
        file_path: Path to the data file

    Returns:
        Dictionary with analysis results
    """
    # Get file information
    try:
        file_stats = os.stat(file_path)
        file_size = file_stats.st_size
        file_name = os.path.basename(file_path)

        file_info: Dict[str, Any] = {
            "Path": file_path,
            "Name": file_name,
            "Size": f"{file_size / 1024:.2f} KB"
            if file_size < 1024 * 1024
            else f"{file_size / (1024 * 1024):.2f} MB",
        }

        # Determine file type by extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        # Process based on file type
        if ext in [".pkl", ".pickle"]:
            try:
                # For pickle files, analyze securely in E2B sandbox
                log.info(
                    "Analyzing pickle file securely in E2B sandbox",
                    file_path=file_path,
                    file_name=file_name,
                    file_size=file_size,
                )

                # Debug: Log the analyze_pickle_files function to ensure it exists
                log.debug(
                    "Type of analyze_pickle_files:",
                    type=str(type(analyze_pickle_files)),
                )

                # Create a string buffer to capture the output of analyze_pickle_files
                output_buffer = io.StringIO()
                original_stdout = sys.stdout
                sys.stdout = output_buffer

                # Call the function from pickle_analyzer to analyze the pickle in a sandbox
                log.debug("About to call analyze_pickle_files", file_path=file_path)
                analyze_pickle_files(
                    [file_path], verbose=True
                )  # Set verbose to True for more output
                log.debug("Finished calling analyze_pickle_files")

                # Restore stdout and get the captured output
                sys.stdout = original_stdout
                analysis_output = output_buffer.getvalue()

                # Debug: Log the raw output
                log.debug(
                    "Raw output from analyze_pickle_files",
                    output_length=len(analysis_output),
                    output_preview=analysis_output[:500]
                    + ("..." if len(analysis_output) > 500 else ""),
                )

                # We're going to parse the output directly without trying to match filenames
                file_info["Type"] = "Pickle File (Securely Analyzed)"
                file_info["Analysis Method"] = "E2B Sandbox"

                # Parse the output and associate all objects with this file
                analyzed_objects = parse_sandbox_output(analysis_output)

                if analyzed_objects:
                    # Add the object information to our results
                    file_info["Analyzed Objects"] = analyzed_objects
                    file_info["Num Objects"] = len(analyzed_objects)

                    # Extract a summary of object types
                    object_type_summary: Dict[str, int] = {}
                    for obj in analyzed_objects:
                        obj_type = obj.get("Object Type", "Unknown")
                        if obj_type in object_type_summary:
                            object_type_summary[obj_type] += 1
                        else:
                            object_type_summary[obj_type] = 1
                    file_info["Object Type Summary"] = object_type_summary

                    log.info(
                        "Successfully analyzed pickle file",
                        file_name=file_name,
                        num_objects=len(analyzed_objects),
                        object_types=object_type_summary,
                    )
                else:
                    log.warning(
                        "No objects found in pickle file analysis",
                        file_name=file_name,
                        analysis_output_sample=analysis_output[:200] + "..."
                        if analysis_output
                        else "No output",
                    )
                    file_info["Warning"] = "No objects found in analysis"
                    file_info["Raw Output"] = analysis_output

                return file_info
            except Exception as e:
                log.error(
                    "Failed to analyze pickle file in sandbox",
                    error=str(e),
                    traceback=traceback.format_exc(),
                    file_path=file_path,
                )
                return {
                    "error": f"Failed to analyze pickle file securely: {str(e)}",
                    "Type": "Pickle File",
                    "Analysis Method": "Failed E2B Sandbox Analysis",
                    "Path": file_path,
                    "Name": file_name,
                    "Size": file_info["Size"],
                }

        elif ext == ".csv":
            try:
                df = pd.read_csv(file_path, low_memory=False)
                file_info.update(analyze_dataframe(df))
                file_info["Type"] = "DataFrame (CSV)"
                return file_info
            except Exception as e:
                log.error(
                    "Failed to analyze CSV file", error=str(e), file_path=file_path
                )
                return {"error": f"Failed to analyze CSV file: {str(e)}"}
        elif ext in {".xls", ".xlsx"}:
            try:
                df = pd.read_excel(file_path)
                file_info.update(analyze_dataframe(df))
                file_info["Type"] = "DataFrame (XLSX)"
                return file_info
            except Exception as e:
                log.error(
                    "Failed to analyze XLSX file", error=str(e), file_path=file_path
                )
                return {"error": f"Failed to analyze Excel file: {str(e)}"}

        elif ext == ".parquet":
            try:
                df = pd.read_parquet(file_path)
                file_info.update(analyze_dataframe(df))
                file_info["Type"] = "DataFrame (Parquet)"
                return file_info
            except Exception as e:
                log.error(
                    "Failed to analyze Parquet file", error=str(e), file_path=file_path
                )
                return {"error": f"Failed to analyze Parquet file: {str(e)}"}

        elif ext in [".json", ".geojson"]:
            try:
                with open(file_path, "r") as f:
                    json_data = json.load(f)

                # Basic information about the JSON file
                file_info["Type"] = "GeoJSON" if ext == ".geojson" else "JSON"

                # Analyze the structure of the JSON
                if isinstance(json_data, dict):
                    file_info["Structure"] = "Dictionary"
                    file_info["Number of keys"] = len(json_data)
                    file_info["Top-level keys"] = list(json_data.keys())[
                        :10
                    ]  # Limit to first 10 keys

                    # Check for GeoJSON structure
                    if ext == ".geojson" or "type" in json_data:
                        json_type = json_data.get("type", "")
                        file_info["GeoJSON Type"] = json_type

                        # Handle different GeoJSON types
                        if json_type == "FeatureCollection" and "features" in json_data:
                            features = json_data["features"]
                            file_info["Features Count"] = len(features)

                            # Analyze a sample feature
                            if features and len(features) > 0:
                                sample_feature = features[0]
                                if "geometry" in sample_feature:
                                    file_info["Geometry Type"] = sample_feature[
                                        "geometry"
                                    ].get("type", "Unknown")

                                if "properties" in sample_feature:
                                    props = sample_feature["properties"]
                                    file_info["Property Keys"] = (
                                        list(props.keys())
                                        if isinstance(props, dict)
                                        else "No properties"
                                    )

                        elif json_type in ["Feature"]:
                            if "geometry" in json_data:
                                file_info["Geometry Type"] = json_data["geometry"].get(
                                    "type", "Unknown"
                                )

                            if "properties" in json_data:
                                props = json_data["properties"]
                                file_info["Property Keys"] = (
                                    list(props.keys())
                                    if isinstance(props, dict)
                                    else "No properties"
                                )

                elif isinstance(json_data, list):
                    file_info["Structure"] = "List"
                    file_info["Number of items"] = len(json_data)

                    # Analyze the first few items
                    if json_data:
                        sample_item = json_data[0]
                        if isinstance(sample_item, dict):
                            file_info["Item Type"] = "Dictionary"
                            if len(json_data) > 0:
                                file_info["Sample Item Keys"] = list(sample_item.keys())
                        else:
                            file_info["Item Type"] = type(sample_item).__name__

                # If it's a structure that can be converted to a DataFrame
                try:
                    # For GeoJSON, extract properties as a DataFrame
                    if (
                        ext == ".geojson"
                        and isinstance(json_data, dict)
                        and "features" in json_data
                    ):
                        # Extract properties from features
                        properties_list = []
                        for feature in json_data["features"]:
                            if "properties" in feature and isinstance(
                                feature["properties"], dict
                            ):
                                properties_list.append(feature["properties"])

                        if properties_list:
                            df = pd.DataFrame(properties_list)
                            df_info = analyze_dataframe(df)
                            file_info["Properties DataFrame"] = df_info
                    # For regular JSON that's a list of dictionaries
                    elif (
                        isinstance(json_data, list)
                        and json_data
                        and isinstance(json_data[0], dict)
                    ):
                        df = pd.DataFrame(json_data)
                        df_info = analyze_dataframe(df)
                        file_info["DataFrame Representation"] = df_info
                except Exception as df_err:
                    log.warning(
                        "Failed to convert JSON to DataFrame",
                        error=str(df_err),
                        file_path=file_path,
                    )

                return file_info
            except Exception as e:
                log.error(
                    f"Failed to analyze {'GeoJSON' if ext == '.geojson' else 'JSON'} file",
                    error=str(e),
                    file_path=file_path,
                )
                return {
                    "error": f"Failed to analyze {'GeoJSON' if ext == '.geojson' else 'JSON'} file: {str(e)}"
                }

        else:
            return {"error": f"Unsupported file type: {ext}"}

    except Exception as e:
        log.error("Failed to analyze data file", error=str(e), file_path=file_path)
        return {"error": f"Failed to analyze data file: {str(e)}"}


def analyze_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze a pandas DataFrame and return its properties.

    Args:
        df: Pandas DataFrame to analyze

    Returns:
        Dictionary with DataFrame properties
    """
    result: Dict[str, Any] = {}

    # Basic DataFrame info
    result["Shape"] = str(df.shape)
    result["Columns"] = list(df.columns)
    result["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # Missing values
    missing_values = df.isnull().sum()
    result["Missing Values"] = {
        col: int(missing) for col, missing in missing_values.items() if missing > 0
    }

    # Sample data - Convert DataFrame to JSON-serializable format
    try:
        # First convert any Timestamp or datetime objects to strings
        sample_df = df.head(5).copy()

        # Convert all datetime columns to strings
        for col in sample_df.select_dtypes(
            include=["datetime64", "datetime64[ns]"]
        ).columns:
            sample_df[col] = sample_df[col].astype(str)

        # For any remaining Timestamp objects or other non-serializable types
        sample_data = sample_df.to_dict(orient="records")

        # Custom conversion for any remaining non-serializable objects
        for row in sample_data:
            for key, value in row.items():
                # Check for pandas Timestamp
                if hasattr(value, "timestamp") and callable(
                    getattr(value, "timestamp")
                ):
                    row[key] = str(value)
                # Handle numpy types
                elif isinstance(value, (np.integer, np.floating)):
                    row[key] = (
                        float(value) if isinstance(value, np.floating) else int(value)
                    )
                elif isinstance(value, np.ndarray):
                    row[key] = value.tolist()
                # Handle any other non-serializable types
                elif not isinstance(
                    value, (str, int, float, bool, list, dict, type(None))
                ):
                    row[key] = str(value)

        # Store sample data with consistent key name
        result["Sample Data"] = sample_data
    except Exception as e:
        # If serialization still fails, provide a simplified version
        log.error(f"Error serializing sample data: {str(e)}")
        result["Sample Data"] = []

    # Column statistics
    column_stats: Dict[str, Dict[str, Any]] = {}

    for col in df.columns:
        stats: Dict[str, Any] = {}
        dtype = str(df[col].dtype)

        if pd.api.types.is_numeric_dtype(df[col]):
            # For numeric columns
            stats["mean"] = (
                float(df[col].mean()) if not pd.isna(df[col].mean()) else None
            )
            stats["std"] = float(df[col].std()) if not pd.isna(df[col].std()) else None
            stats["min"] = float(df[col].min()) if not pd.isna(df[col].min()) else None
            stats["max"] = float(df[col].max()) if not pd.isna(df[col].max()) else None
            stats["median"] = (
                float(df[col].median()) if not pd.isna(df[col].median()) else None
            )

            # Only add if not all values are NaN
            if any(v is not None for v in stats.values()):
                column_stats[str(col)] = stats

        elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(
            df[col]
        ):
            # For string/object columns
            value_counts = df[col].value_counts(dropna=False).head(5).to_dict()

            # Convert any non-serializable keys or values
            value_counts_serializable = {}
            for k, v in value_counts.items():
                # Convert key if it's a Timestamp or other non-serializable type
                key = str(k) if not isinstance(k, (str, int, float, bool)) else k
                # Convert value to int (should be a count)
                val = int(v)
                value_counts_serializable[key] = val

            stats["unique_count"] = df[col].nunique()
            stats["top_values"] = value_counts_serializable

            column_stats[str(col)] = stats

    result["Column Stats"] = column_stats

    return result


def extract_text_from_file(file_path: str) -> Dict[str, Any]:
    """
    Extract text content from various file types.

    Args:
        file_path: Path to the context file

    Returns:
        Dictionary with extracted text and possibly error messages
    """
    file_name = os.path.basename(file_path)
    file_stats = os.stat(file_path)
    file_size = file_stats.st_size

    result: Dict[str, Any] = {
        "path": file_path,
        "name": file_name,
        "size": f"{file_size / 1024:.2f} KB"
        if file_size < 1024 * 1024
        else f"{file_size / (1024 * 1024):.2f} MB",
    }

    # Determine file type by extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    text_content = ""

    try:
        if ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                text_content = f.read()
            result["type"] = "Text File"

        elif ext == ".pdf":
            try:
                import PyPDF2

                with open(file_path, "rb") as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    for page in pdf_reader.pages:
                        text_content += page.extract_text() + "\n\n"
                result["type"] = "PDF File"
            except ImportError:
                result["error"] = "PyPDF2 library not installed. Unable to process PDF."
                log.error("PyPDF2 library not installed")

        elif ext in [".docx", ".doc"]:
            try:
                import docx

                doc = docx.Document(file_path)
                for para in doc.paragraphs:
                    text_content += para.text + "\n"
                result["type"] = "Word Document"
            except ImportError:
                result["error"] = (
                    "python-docx library not installed. Unable to process Word document."
                )
                log.error("python-docx library not installed")
        else:
            result["error"] = f"Unsupported file type: {ext}"
            return result

        # Truncate text content if too large
        if len(text_content) > 1000000:  # ~1MB of text
            text_content = text_content[:1000000] + "... [truncated]"
            result["warning"] = "Text content was truncated due to large size"

        result["text"] = text_content
        # For backward compatibility
        result["Text"] = text_content
        return result

    except Exception as e:
        log.error("Failed to extract text from file", error=str(e), file_path=file_path)
        result["error"] = f"Failed to extract text: {str(e)}"
        return result


def correlate_data_files(data_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Find correlations between data files

    Args:
        data_results: Dictionary of data file analysis results

    Returns:
        Dictionary with correlation results
    """
    log.info("Argument", arg=data_results)
    if not data_results or len(data_results) < 2:
        log.warning(
            "Not enough data files for correlation analysis",
            count=len(data_results) if data_results else 0,
        )
        return {"error": "Need at least two data files to find correlations"}

    correlations: Dict[str, Any] = {
        "shared_columns": {},
        "common_data_types": {},
        "similar_sizes": [],
        "potential_joins": [],
        "llm_suggested_joins": [],  # New field for LLM-identified join keys
    }

    # Find common columns across DataFrames
    dataframes = {
        fname: info
        for fname, info in data_results.items()
        if info.get("Type", "").startswith("DataFrame")
    }

    log.info(
        "DataFrame files identified for correlation",
        count=len(dataframes),
        filenames=list(dataframes.keys()),
    )

    # Exit early if we don't have at least 2 DataFrames
    if len(dataframes) < 2:
        log.warning(
            "Not enough DataFrame files for correlation analysis",
            dataframes_found=len(dataframes),
        )
        return {
            "message": "Not enough DataFrame files found for correlation analysis",
            "dataframes_found": len(dataframes),
        }

    # Track all columns seen
    all_columns: Dict[str, List[str]] = {}

    # Find shared columns
    for fname, info in dataframes.items():
        columns = info.get("Columns", [])
        log.debug(f"Processing columns for file {fname}", column_count=len(columns))
        for col in columns:
            col_str = str(col)
            if col_str not in all_columns:
                all_columns[col_str] = []
            all_columns[col_str].append(fname)

    # Filter for columns that appear in multiple files
    correlations["shared_columns"] = {
        col: files for col, files in all_columns.items() if len(files) > 1
    }
    log.info("Shared columns identified", count=len(correlations["shared_columns"]))

    # Find potential join keys
    potential_joins: List[Dict[str, Any]] = []
    for col, files in correlations["shared_columns"].items():
        if len(files) >= 2:
            potential_joins.append({"column": col, "files": files})
    correlations["potential_joins"] = potential_joins
    log.info("Potential join keys identified", count=len(potential_joins))

    # Find similar data types
    common_types_count = 0
    for fname1, info1 in dataframes.items():
        for fname2, info2 in dataframes.items():
            if fname1 >= fname2:  # Skip self-comparisons and duplicates
                continue

            common_cols = set(info1.get("Columns", [])).intersection(
                set(info2.get("Columns", []))
            )
            if common_cols:
                dtypes1 = info1.get("dtypes", {})
                dtypes2 = info2.get("dtypes", {})

                matching_dtypes = {}
                for col in common_cols:
                    col_str = str(col)
                    if col_str in dtypes1 and col_str in dtypes2:
                        if dtypes1[col_str] == dtypes2[col_str]:
                            matching_dtypes[col_str] = dtypes1[col_str]

                if matching_dtypes:
                    key = f"{fname1} ⟷ {fname2}"
                    if "common_data_types" not in correlations:
                        correlations["common_data_types"] = {}
                    correlations["common_data_types"][key] = matching_dtypes
                    common_types_count += 1

    log.info("Common data types identified", count=common_types_count)

    # Find files with similar sizes
    sizes = []
    for fname, info in dataframes.items():
        shape_str = info.get("Shape", "(0, 0)")
        # Handle shape stored as string like "(100, 5)"
        if (
            isinstance(shape_str, str)
            and shape_str.startswith("(")
            and shape_str.endswith(")")
        ):
            try:
                # Parse the row count from the shape string
                row_count = int(shape_str.strip("()").split(",")[0].strip())
                sizes.append((fname, row_count))
            except (ValueError, IndexError):
                continue
        elif isinstance(shape_str, tuple) and len(shape_str) == 2:
            # Handle shape stored as actual tuple
            sizes.append((fname, shape_str[0]))

    sizes.sort(key=lambda x: x[1])

    similar_sizes: List[Dict[str, Any]] = []
    for i in range(len(sizes) - 1):
        curr_file, curr_size = sizes[i]
        next_file, next_size = sizes[i + 1]

        # If within 10% of each other
        if curr_size > 0 and abs(curr_size - next_size) / curr_size < 0.1:
            similar_sizes.append(
                {"files": [curr_file, next_file], "row_counts": [curr_size, next_size]}
            )
    correlations["similar_sizes"] = similar_sizes
    log.info("Similar sized files identified", count=len(similar_sizes))

    # NEW PART: Use LLM to identify potential join keys based on column names and sample values
    if openai_client:
        log.info("Starting LLM-based join key analysis")
        try:
            # Prepare detailed information about each dataframe for the LLM
            dataframe_details = {}
            missing_sample_data = []

            for fname, info in dataframes.items():
                # Extract column information
                columns = info.get("Columns", [])

                # Extract sample data if available - check different possible keys
                sample_data = []

                # Try different keys that might contain sample data
                sample_data_found = False
                sample_key_used = None

                for key in ["Sample Data", "sample_data", "SampleData"]:
                    if key in info and info[key]:
                        raw_sample = info[key]
                        log.debug(
                            f"Found sample data in key '{key}' for file {fname}",
                            is_list=isinstance(raw_sample, list),
                            is_str=isinstance(raw_sample, str),
                        )

                        # Handle different formats
                        if isinstance(raw_sample, list):
                            sample_data = raw_sample
                            sample_data_found = True
                            sample_key_used = key
                            break
                        elif isinstance(raw_sample, str):
                            # Try to parse JSON string
                            try:
                                if raw_sample.startswith("["):
                                    sample_data = json.loads(
                                        raw_sample.replace("'", '"')
                                    )
                                    sample_data_found = True
                                    sample_key_used = key
                                    break
                                elif raw_sample.startswith("{"):
                                    # Might be a dict format with columns as keys
                                    sample_dict = json.loads(
                                        raw_sample.replace("'", '"')
                                    )
                                    # Convert to records format if possible
                                    if all(
                                        isinstance(sample_dict[col], list)
                                        for col in sample_dict
                                    ):
                                        # Get the length of the first list
                                        first_col = list(sample_dict.keys())[0]
                                        length = len(sample_dict[first_col])
                                        # Create records
                                        records = []
                                        for i in range(length):
                                            record = {}
                                            for col, values in sample_dict.items():
                                                if i < len(values):
                                                    record[col] = values[i]
                                            records.append(record)
                                        sample_data = records
                                        sample_data_found = True
                                        sample_key_used = key
                                        break
                            except Exception as e:
                                log.warning(
                                    f"Failed to parse sample data JSON for file {fname}",
                                    error=str(e),
                                    key=key,
                                )
                                # Failed to parse, continue to next key
                                pass

                if not sample_data_found:
                    missing_sample_data.append(fname)
                    log.warning(f"No sample data found for file {fname}")
                else:
                    log.info(
                        f"Sample data found for file {fname}",
                        key=sample_key_used,
                        records_count=len(sample_data),
                    )

                # Create a summary of sample values for each column
                column_samples = {}
                for col in columns:
                    col_str = str(col)
                    col_values = []

                    # Extract values from sample data
                    for row in sample_data[:3]:  # Use up to 3 rows
                        if isinstance(row, dict) and col_str in row:
                            value = row[col_str]
                            # Convert non-string values to strings
                            if not isinstance(value, str):
                                value = str(value)
                            col_values.append(value)

                    if col_values:
                        column_samples[col_str] = col_values

                # Add to dataframe details
                dataframe_details[fname] = {
                    "columns": columns,
                    "column_samples": column_samples,
                    "shape": info.get("Shape", "Unknown"),
                    "dtypes": info.get("dtypes", {}),
                }

            log.info(
                "Prepared dataframe details for LLM",
                dataframe_count=len(dataframe_details),
                missing_sample_data=missing_sample_data,
            )

            # Prepare the prompt for the LLM
            prompt = f"""
You are analyzing multiple dataframes to identify potential join keys. I'll provide details about each dataframe including column names and sample values.

Your task is to identify columns across different dataframes that could potentially be used as join keys, even if they have different column names. 
Look for columns that might contain the same type of entity identifiers, foreign keys, or related information.

Consider:
1. Columns with similar names or semantically related names (e.g. 'user_id' and 'uid')
2. Columns with similar value patterns
3. Columns that might represent the same entities across different tables

Here are the dataframes:
{json.dumps(dataframe_details, indent=2)}

Respond with a JSON array where each item represents a potential join relationship. Each item should be an object with these fields:
- "source_file": String, name of the first dataframe
- "source_column": String, column name in the first dataframe
- "target_file": String, name of the second dataframe
- "target_column": String, column name in the second dataframe
- "confidence": Number between 0 and 1, your confidence in this relationship
- "explanation": String, brief explanation of why these columns might be joinable

Only include column pairs that have a reasonable chance of being related. Focus on quality over quantity.
If you're not confident about any join keys, return an empty array [].
"""

            log.info(
                "Sending prompt to OpenAI for join key analysis",
                prompt_length=len(prompt),
                dataframe_count=len(dataframe_details),
            )

            # Call the OpenAI API
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data analysis expert that identifies potential relationships between dataframes.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,  # Lower temperature for more conservative suggestions
            )

            # Extract and parse the response
            llm_response = response.choices[0].message.content
            if llm_response:
                log.info(
                    "Received response from OpenAI",
                    response_length=len(llm_response),
                    response_snippet=llm_response[:100],
                )

            # Log the full response for debugging
            log.debug("Full LLM response", response=llm_response)

            # Parse the JSON response
            if llm_response:
                try:
                    suggested_joins = json.loads(llm_response)
                    # Log the raw parsed JSON
                    log.debug(
                        "Parsed LLM response JSON",
                        is_dict=isinstance(suggested_joins, dict),
                        is_list=isinstance(suggested_joins, list),
                    )

                    # Check if it's a list or if it's wrapped in another object
                    if isinstance(suggested_joins, dict) and "joins" in suggested_joins:
                        suggested_joins = suggested_joins["joins"]
                        log.info("Extracted 'joins' array from response dictionary")
                    elif isinstance(suggested_joins, list):
                        # Already in the right format
                        log.info("LLM response is already in list format")
                    elif isinstance(suggested_joins, dict):
                        # Look for any array in the response
                        found_array = False
                        for key, value in suggested_joins.items():
                            if isinstance(value, list):
                                log.info(
                                    f"Found array in key '{key}' of response dictionary"
                                )
                                suggested_joins = value
                                found_array = True
                                break
                        if not found_array:
                            # No array found, create empty list
                            log.warning(
                                "No array found in response dictionary, using empty list"
                            )
                            suggested_joins = []
                    else:
                        # Not a recognized format, create empty list
                        log.warning(
                            "Unrecognized format in LLM response, using empty list",
                            type=type(suggested_joins).__name__,
                        )
                        suggested_joins = []

                    # Ensure it's a list
                    if not isinstance(suggested_joins, list):
                        log.warning(
                            "Ensuring suggested_joins is a list",
                            original_type=type(suggested_joins).__name__,
                        )
                        suggested_joins = []

                    correlations["llm_suggested_joins"] = suggested_joins
                    # Add a flag to indicate the LLM was successfully called but found no joins
                    if len(suggested_joins) == 0:
                        log.info("LLM found no join keys")
                        correlations["llm_no_joins_found"] = True

                    log.info(
                        "LLM suggested join keys analysis complete",
                        count=len(suggested_joins),
                        keys=[
                            (
                                j.get("source_file", "?"),
                                j.get("source_column", "?"),
                                j.get("target_file", "?"),
                                j.get("target_column", "?"),
                            )
                            for j in suggested_joins[:5]
                        ],
                    )  # Log first 5 joins
                except json.JSONDecodeError as e:
                    log.error(
                        "Failed to parse LLM response as JSON",
                        error=str(e),
                        response=llm_response[:200],
                    )
                    correlations["llm_analysis_error"] = f"JSON parse error: {str(e)}"
        except Exception as e:
            log.error(
                "Error in LLM-based join key analysis",
                error=str(e),
                traceback=traceback.format_exc(),
            )
            # Don't fail the whole function if LLM analysis fails
            correlations["llm_analysis_error"] = str(e)
    else:
        log.warning("OpenAI client not available, skipping LLM-based join key analysis")

    log.info(
        "Correlation analysis complete",
        shared_columns=len(correlations.get("shared_columns", {})),
        potential_joins=len(correlations.get("potential_joins", [])),
        similar_sizes=len(correlations.get("similar_sizes", [])),
        common_data_types=len(correlations.get("common_data_types", {})),
        llm_suggested_joins=len(correlations.get("llm_suggested_joins", [])),
    )

    return correlations


def generate_analysis_plan(
    data_results: Dict[str, Dict[str, Any]],
    context_texts: Dict[str, str],
    correlations: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate an analysis plan using OpenAI API

    Args:
        data_results: Dictionary of data file analysis results
        context_texts: Dictionary of context file texts
        correlations: Dictionary of correlation results

    Returns:
        Dictionary with analysis plan and generated code
    """
    if not openai_client:
        return {"error": "OpenAI client not initialized. Please check your API key."}

    # Prepare the prompt
    data_summary = json.dumps(
        {
            k: {
                "type": v.get("Type", "Unknown"),
                "shape": v.get("Shape", "Unknown"),
                "columns": v.get("Columns", [])[:10],  # Limit columns for prompt size
            }
            for k, v in data_results.items()
        },
        indent=2,
    )

    # Prepare context text (truncate if too long)
    context_summary = ""
    for filename, text in context_texts.items():
        # Truncate text if too long
        if len(text) > 500:
            text = text[:500] + "... [truncated]"
        context_summary += f"=== {filename} ===\n{text}\n\n"

    correlation_summary = json.dumps(correlations, indent=2)

    prompt = f"""As a data science assistant, analyze the following datasets and context information to create an analysis plan.

DATA FILES:
{data_summary}

CORRELATIONS BETWEEN DATA FILES:
{correlation_summary}

CONTEXT FROM TEXT FILES:
{context_summary}

Based on this information, please:
1. Create a concise analysis plan
2. Generate Python code that could be executed in an E2B sandbox to analyze these datasets
3. The code should focus on finding patterns, correlations, and insights between the datasets
4. Utilize any context information to guide the analysis
5. Include visualizations where appropriate

Respond with a JSON object containing:
- "analysis_plan": A step-by-step plan for data analysis
- "analysis_code": Complete Python code that could be executed

In "analysis_code":
- The code should be self-contained and handle reading the files from their paths.
- It should contain no placeholders or truncations, only valid Python.
- Avoid triple backticks at the start or the end, just issue Python straightaway."""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            seed=SEED,
            messages=[
                {
                    "role": "system",
                    "content": "You are a data science assistant that generates analysis plans and Python code.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )

        # Extract the response
        response_text = response.choices[0].message.content
        if response_text:
            result = json.loads(response_text)

            # Add timestamp
            result["timestamp"] = pd.Timestamp.now().isoformat()

            return result
        else:
            return {"error": "No response received from OpenAI API"}

    except Exception as e:
        return {
            "error": f"Failed to generate analysis plan: {str(e)}",
            "traceback": traceback.format_exc(),
        }


def parse_sandbox_output(analysis_text: str) -> List[Dict[str, Any]]:
    """
    Parse the raw output from the E2B sandbox analysis of pickle files.

    Instead of trying to associate each section with a filename, we treat each section
    as an individual object analysis and return a list of analyzed objects.

    Args:
        analysis_text: The raw text output from pickle_analyzer

    Returns:
        List of dictionaries with analysis results for each object
    """
    # Debug log the analysis text
    log.debug(
        "Parsing sandbox output",
        text_length=len(analysis_text),
        text_sample=analysis_text[:200] + ("..." if len(analysis_text) > 200 else ""),
    )

    # Split into sections by the separator (50 equal signs)
    sections = analysis_text.split("=" * 50)
    sections = [s.strip() for s in sections if s.strip()]

    log.debug(f"Found {len(sections)} sections in analysis text")

    analyzed_objects = []

    for i, section in enumerate(sections):
        lines = section.split("\n")
        if len(lines) < 2:
            log.warning(f"Section {i} too short to parse", lines_count=len(lines))
            continue

        # Extract information about this object
        object_info = {}
        current_key = None
        current_value = []

        # Try to extract object type directly from the first few lines
        object_type = "Unknown"
        for idx, line in enumerate(lines[:10]):  # Check first 10 lines
            if "Object Type:" in line:
                try:
                    object_type = line.split("Object Type:")[1].strip()
                    break
                except:
                    pass
            elif "Type:" in line:
                try:
                    object_type = line.split("Type:")[1].strip()
                    break
                except:
                    pass

        # Check for specific types in output
        if object_type == "Unknown":
            # Look for specific patterns in text
            if any("dict" in line.lower() for line in lines[:20]):
                object_type = "dict"
            elif any("list" in line.lower() for line in lines[:20]):
                object_type = "list"
            elif any("dataframe" in line.lower() for line in lines[:20]):
                object_type = "DataFrame"
            elif any("ndarray" in line.lower() for line in lines[:20]):
                object_type = "ndarray"

        # Explicitly set the Object Type
        object_info["Object Type"] = object_type

        # Check if this section contains object analysis
        contains_analysis = False
        for line in lines:
            if (
                "Object Type:" in line
                or "Type:" in line
                or "Value:" in line
                or "Contents:" in line
                or "Length:" in line
                or "Shape:" in line
                or "Size:" in line
                or "keys_count" in line
                or "Number of keys:" in line
            ):
                contains_analysis = True
                break

        if not contains_analysis:
            log.debug(
                f"Section {i} doesn't appear to contain object analysis, skipping"
            )
            continue

        # Process line by line
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # If it's a new key-value pair
            if ": " in line and not line.startswith(" "):
                # Save the previous key-value pair if it exists
                if current_key:
                    object_info[current_key] = (
                        "\n".join(current_value)
                        if len(current_value) > 1
                        else current_value[0]
                    )
                    current_value = []

                # Start a new key-value pair
                try:
                    current_key, value = line.split(": ", 1)

                    # Normalize key names for consistency with UI expectations
                    if current_key.lower() == "type":
                        current_key = "Object Type"
                        # Update the object_type variable as well
                        object_type = value
                    elif current_key.lower() == "size":
                        current_key = "Size (bytes)"
                    elif current_key.lower() == "length":
                        current_key = "Length"
                    elif current_key.lower() == "shape":
                        current_key = "Shape"
                    elif current_key.lower() == "number of keys":
                        current_key = "Number of Keys"

                    current_value.append(value)
                except ValueError:
                    # If there's a problem with splitting, just store the whole line
                    log.warning(f"Could not parse line properly: {line}")
                    if current_key:
                        current_value.append(line)
            else:
                # Continue with the current value
                if current_key:
                    current_value.append(line)

        # Save the last key-value pair
        if current_key:
            object_info[current_key] = (
                "\n".join(current_value) if len(current_value) > 1 else current_value[0]
            )

        # Make sure Object Type is set
        if "Object Type" not in object_info:
            object_info["Object Type"] = object_type

        # Add raw section text for debugging
        object_info["Raw Section Text"] = (
            section[:500] + "..." if len(section) > 500 else section
        )

        if object_info:
            # Add an index to identify the object
            object_info["Object Index"] = str(i)
            analyzed_objects.append(object_info)
            log.debug(f"Added object #{i} with {len(object_info)} properties")

    # Calculate object type summary
    object_type_summary: Dict[str, int] = {}
    for obj in analyzed_objects:
        obj_type = obj.get("Object Type", "Unknown")
        if obj_type in object_type_summary:
            object_type_summary[obj_type] += 1
        else:
            object_type_summary[obj_type] = 1

    log.info(f"Finished parsing {len(analyzed_objects)} objects from sandbox output")
    log.info(f"Object type summary: {object_type_summary}")

    return analyzed_objects


def main():
    """Run the Dash app server"""
    app.run(debug=True)


if __name__ == "__main__":
    main()
