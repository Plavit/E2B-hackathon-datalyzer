import os
import sys
import json
import base64
import pickle
import traceback
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple
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
from PyPDF2 import PdfReader
from docx import Document
import openai
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

# Initialize OpenAI client (will use API key from environment)
try:
    openai_client: Optional[OpenAI] = OpenAI()
except Exception as e:
    log.error("Failed to initialize OpenAI client", error=str(e))
    openai_client = None  # type: ignore

# Define application layout
layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H1(
                            "Enhanced Data Analysis Platform",
                            className="text-center my-4",
                        ),
                        html.P(
                            "Upload data files (Pickle, CSV, Parquet) and context files (Text, PDF, Word) for comprehensive analysis",
                            className="text-center lead mb-4",
                        ),
                    ],
                    width=12,
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.H5(
                                            "Upload Data Files", className="card-title"
                                        ),
                                        html.P(
                                            "Upload data files (.pkl, .pickle, .csv, .parquet) for analysis",
                                            className="card-text text-muted",
                                        ),
                                        dcc.Upload(
                                            id="upload-data",
                                            children=html.Div(
                                                [
                                                    html.I(
                                                        className="fas fa-upload me-2"
                                                    ),
                                                    "Drag and Drop or ",
                                                    html.A("Select Data Files"),
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
                    md=6,
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.H5(
                                            "Upload Context Files",
                                            className="card-title",
                                        ),
                                        html.P(
                                            "Upload text, PDF, or Word files to provide context",
                                            className="card-text text-muted",
                                        ),
                                        dcc.Upload(
                                            id="upload-context",
                                            children=html.Div(
                                                [
                                                    html.I(
                                                        className="fas fa-upload me-2"
                                                    ),
                                                    "Drag and Drop or ",
                                                    html.A("Select Context Files"),
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
                                            accept=".txt, .pdf, .docx, .doc",
                                        ),
                                    ]
                                )
                            ],
                            className="mb-4",
                        ),
                    ],
                    md=6,
                ),
            ]
        ),
        # Loading spinner for when analysis is running
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
        # Display files currently being analyzed
        dbc.Row([dbc.Col([html.Div(id="files-being-analyzed")], width=12)]),
        # Error messages
        dbc.Row([dbc.Col([html.Div(id="error-message")], width=12)]),
        # Results section
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(id="data-output", className="mt-4"),
                    ],
                    width=12,
                )
            ]
        ),
        # Context files section
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(id="context-output", className="mt-4"),
                    ],
                    width=12,
                )
            ]
        ),
        # Correlation analysis section
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H3("Correlation Analysis", className="mt-4 mb-3"),
                        html.Button(
                            "Analyze Correlations",
                            id="analyze-correlations-button",
                            className="btn btn-primary mb-3",
                        ),
                        html.Div(id="correlation-output"),
                    ],
                    width=12,
                )
            ]
        ),
        # AI Analysis Plan section
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H3("AI Analysis Plan", className="mt-4 mb-3"),
                        html.Button(
                            "Generate Analysis Plan",
                            id="generate-plan-button",
                            className="btn btn-primary mb-3",
                        ),
                        html.Div(id="ai-plan-output"),
                    ],
                    width=12,
                )
            ]
        ),
        # AI Analysis Results section
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H3("AI Analysis Results", className="mt-4 mb-3"),
                        html.Div(id="ai-results-output"),
                    ],
                    width=12,
                )
            ]
        ),
        # Store components for data
        dcc.Store(id="data-store", storage_type="memory"),
        dcc.Store(id="context-store", storage_type="memory"),
        dcc.Store(id="correlation-store", storage_type="memory"),
        dcc.Store(id="ai-plan-store", storage_type="memory"),
        # Footer
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Hr(),
                        html.P(
                            "Powered by E2B Sandbox and OpenAI - A secure environment for comprehensive data analysis",
                            className="text-center text-muted small",
                        ),
                    ],
                    width=12,
                )
            ],
            className="mt-5",
        ),
    ],
    fluid=True,
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
        "data": [".pkl", ".pickle", ".csv", ".parquet", ".xls", ".xlsx"],
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


def parse_analysis_results(analysis_text: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse the raw analysis text to extract structured information

    Args:
        analysis_text: The raw text output from pickle_analyzer

    Returns:
        Dictionary with parsed analysis results
    """
    # Debug log the analysis text
    log.debug(
        "Parsing analysis results",
        text_length=len(analysis_text),
        text_sample=analysis_text[:200] + ("..." if len(analysis_text) > 200 else ""),
    )

    # Split into sections by file
    sections = analysis_text.split("=" * 50)
    sections = [s.strip() for s in sections if s.strip()]

    log.debug(f"Found {len(sections)} sections in analysis text")

    results = {}

    for i, section in enumerate(sections):
        lines = section.split("\n")
        if len(lines) < 2:
            log.warning(f"Section {i} too short to parse", lines_count=len(lines))
            continue

        # Extract filename from the first line
        first_line = lines[0]
        log.debug(f"Processing section {i}, first line: {first_line}")

        # Check for the expected "Analysis of [filename]:" pattern
        if "Analysis of " in first_line and ":" in first_line:
            file_name = first_line.replace("Analysis of ", "").replace(":", "").strip()
            log.debug(f"Extracted file name: '{file_name}'")
        else:
            log.warning(
                f"Couldn't extract file name from section {i}", first_line=first_line
            )
            # Try a more flexible pattern
            file_name = f"unknown_file_{i}"
            for line in lines[:3]:  # Check first few lines
                if ".pkl" in line or ".pickle" in line:
                    potential_name = line.split("/")[-1].split()[0]
                    if potential_name.endswith(".pkl") or potential_name.endswith(
                        ".pickle"
                    ):
                        file_name = potential_name
                        log.debug(f"Found potential file name: {file_name}")
                        break

        # Extract other information
        file_info = {}
        current_key = None
        current_value = []

        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue

            # If it's a new key-value pair
            if ": " in line and not line.startswith(" "):
                # Save the previous key-value pair if it exists
                if current_key:
                    file_info[current_key] = (
                        "\n".join(current_value)
                        if len(current_value) > 1
                        else current_value[0]
                    )
                    current_value = []

                # Start a new key-value pair
                current_key, value = line.split(": ", 1)
                current_value.append(value)
            else:
                # Continue with the current value
                if current_key:
                    current_value.append(line)

        # Save the last key-value pair
        if current_key:
            file_info[current_key] = (
                "\n".join(current_value) if len(current_value) > 1 else current_value[0]
            )

        results[file_name] = file_info
        log.debug(f"Added results for '{file_name}' with {len(file_info)} properties")

    log.info(
        f"Finished parsing results for {len(results)} files",
        file_names=list(results.keys()),
    )
    return results


def create_file_summary_card(file_name: str, file_info: Dict[str, Any]) -> dbc.Card:
    """
    Create a card with a summary of the file analysis

    Args:
        file_name: Name of the file
        file_info: Dictionary with file information

    Returns:
        A Dash Bootstrap Components Card
    """
    # Extract file type for additional component generation
    file_type = file_info.get("Type", "Unknown")

    # Create list group items dynamically from file_info
    list_group_items = []
    for key, value in file_info.items():
        # Skip complex nested data that would be better displayed in the additional info section
        if isinstance(value, (dict, list, tuple)) or key in [
            "sample_keys",
            "sample_elements",
            "sample_data",
            "columns",
            "dtypes",
        ]:
            continue

        # Format the value as string if it's not already
        if not isinstance(value, str):
            value = str(value)

        list_group_items.append(
            dbc.ListGroupItem([html.Strong(f"{key}: "), html.Span(value)])
        )

    # Create card content
    card_content = [
        dbc.CardHeader(html.H5(file_name, className="mb-0")),
        dbc.CardBody(
            [
                html.H6("Basic Information", className="card-subtitle mb-2 text-muted"),
                dbc.ListGroup(list_group_items, className="mb-3"),
                # Additional information based on file type
                get_additional_info_component(file_type, file_info),
            ]
        ),
    ]

    return dbc.Card(card_content, className="mb-4")


def get_additional_info_component(
    file_type: str, file_info: Dict[str, Any]
) -> html.Div:
    """
    Create appropriate components to display additional information based on file type

    Args:
        file_type: Type of the file (dict, list, DataFrame, etc.)
        file_info: Dictionary with file information

    Returns:
        Dash HTML component
    """
    components = []

    if "Pickle File" in file_type:
        # Information for pickle file
        analyzed_objects = file_info.get("Analyzed Objects", [])
        if analyzed_objects:
            # Display summary of analyzed objects
            object_count = len(analyzed_objects)
            object_types = file_info.get("Object Type Summary", {})

            components.extend(
                [
                    html.H6("Pickle Analysis Summary", className="mt-3"),
                    dbc.ListGroup(
                        [
                            dbc.ListGroupItem(
                                [
                                    html.Strong("Objects Found: "),
                                    html.Span(str(object_count)),
                                ]
                            ),
                            dbc.ListGroupItem(
                                [
                                    html.Strong("Object Types: "),
                                    html.Span(
                                        ", ".join(
                                            [
                                                f"{k} ({v})"
                                                for k, v in object_types.items()
                                            ]
                                        )
                                    ),
                                ]
                            ),
                        ],
                        className="mb-3",
                    ),
                ]
            )

            # Create cards for each object - directly visible without collapsible panels
            object_cards = []
            for i, obj in enumerate(analyzed_objects):
                obj_type = obj.get("Object Type", "Unknown")

                # Create list items for each property
                obj_details = []
                for key, value in obj.items():
                    if key not in ["Object Index"]:
                        obj_details.append(
                            dbc.ListGroupItem(
                                [html.Strong(f"{key}: "), html.Span(str(value))]
                            )
                        )

                # Create card for this object - directly visible
                object_cards.append(
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                html.H6(
                                    f"Object #{i + 1}: {obj_type}", className="mb-0"
                                )
                            ),
                            dbc.CardBody([dbc.ListGroup(obj_details)]),
                        ],
                        className="mb-3",
                    )
                )

            components.append(
                html.Div(
                    [
                        html.H6("Analyzed Objects", className="mt-3"),
                        html.Div(object_cards),
                    ]
                )
            )
        else:
            # Show raw output if no objects were parsed
            raw_output = file_info.get("Raw Output", "No output available")
            components.append(
                html.Div(
                    [
                        html.H6("Raw Analysis Output", className="mt-3"),
                        dbc.Card(
                            dbc.CardBody(
                                html.Pre(
                                    raw_output,
                                    style={"maxHeight": "300px", "overflowY": "auto"},
                                )
                            ),
                            className="mb-3",
                        ),
                    ]
                )
            )
    elif file_type == "dict":
        # Information for dictionary
        keys_count = file_info.get("Number of keys", "Unknown")
        key_types = file_info.get("Key types", "Unknown")
        value_types = file_info.get("Value types", "Unknown")
        sample_keys = file_info.get("Sample keys", "Unknown")

        components.extend(
            [
                html.H6("Dictionary Details", className="mt-3"),
                dbc.ListGroup(
                    [
                        dbc.ListGroupItem(
                            [html.Strong("Number of keys: "), html.Span(keys_count)]
                        ),
                        dbc.ListGroupItem(
                            [html.Strong("Key types: "), html.Span(key_types)]
                        ),
                        dbc.ListGroupItem(
                            [html.Strong("Value types: "), html.Span(value_types)]
                        ),
                        dbc.ListGroupItem(
                            [html.Strong("Sample keys: "), html.Span(str(sample_keys))]
                        ),
                    ],
                    className="mb-3",
                ),
            ]
        )

        # Create a pie chart of value types if available
        if isinstance(value_types, str) and "," in value_types:
            try:
                value_type_list = [vt.strip() for vt in value_types.split(",")]
                value_counts = {vt: 1 for vt in value_type_list}

                fig = px.pie(
                    names=list(value_counts.keys()),
                    values=list(value_counts.values()),
                    title="Value Types Distribution",
                )

                components.append(dcc.Graph(figure=fig, className="mt-3"))
            except Exception:
                pass

    elif file_type == "list" or file_type == "tuple":
        # Information for list or tuple
        length = file_info.get("Length", "Unknown")
        element_types = file_info.get("Element types", "Unknown")
        sample_elements = file_info.get("Sample elements", "Unknown")

        components.extend(
            [
                html.H6(f"{file_type.capitalize()} Details", className="mt-3"),
                dbc.ListGroup(
                    [
                        dbc.ListGroupItem([html.Strong("Length: "), html.Span(length)]),
                        dbc.ListGroupItem(
                            [html.Strong("Element types: "), html.Span(element_types)]
                        ),
                        dbc.ListGroupItem(
                            [
                                html.Strong("Sample elements: "),
                                html.Span(str(sample_elements)),
                            ]
                        ),
                    ],
                    className="mb-3",
                ),
            ]
        )

    elif file_type == "DataFrame":
        # Information for DataFrame
        shape = file_info.get("Shape", "Unknown")
        columns = file_info.get("Columns", "Unknown")

        components.extend(
            [
                html.H6("DataFrame Details", className="mt-3"),
                dbc.ListGroup(
                    [
                        dbc.ListGroupItem([html.Strong("Shape: "), html.Span(shape)]),
                        dbc.ListGroupItem(
                            [html.Strong("Columns: "), html.Span(str(columns))]
                        ),
                    ],
                    className="mb-3",
                ),
            ]
        )

        # Try to parse and display a sample of the dataframe
        if "Sample data" in file_info:
            components.append(html.H6("Sample Data (First 5 rows):", className="mt-3"))
            try:
                if isinstance(file_info["Sample data"], dict):
                    sample_data = file_info["Sample data"]
                else:
                    # Try to parse from string (simplified)
                    sample_text = file_info["Sample data"]
                    if sample_text.startswith("{") and sample_text.endswith("}"):
                        sample_data = json.loads(sample_text.replace("'", '"'))
                    else:
                        raise ValueError("Cannot parse sample data")

                # Create a table from the sample data
                table_header = [html.Tr([html.Th(col) for col in sample_data.keys()])]

                # Get the number of rows in the first column
                first_col = list(sample_data.values())[0]
                num_rows = len(first_col) if isinstance(first_col, list) else 1

                # Create the table rows
                table_body = []
                for i in range(num_rows):
                    table_body.append(
                        html.Tr(
                            [
                                html.Td(
                                    sample_data[col][i]
                                    if isinstance(sample_data[col], list)
                                    and i < len(sample_data[col])
                                    else ""
                                )
                                for col in sample_data.keys()
                            ]
                        )
                    )

                components.append(
                    dbc.Table(
                        [html.Thead(table_header), html.Tbody(table_body)],
                        bordered=True,
                        hover=True,
                        responsive=True,
                    )
                )
            except Exception:
                components.append(
                    html.P("Failed to parse sample data", className="text-muted")
                )

    elif file_type == "ndarray":
        # Information for NumPy array
        shape = file_info.get("Shape", "Unknown")
        dtype = file_info.get("Data type", "Unknown")
        sample_data = file_info.get("Sample data", "Unknown")

        components.extend(
            [
                html.H6("NumPy Array Details", className="mt-3"),
                dbc.ListGroup(
                    [
                        dbc.ListGroupItem([html.Strong("Shape: "), html.Span(shape)]),
                        dbc.ListGroupItem(
                            [html.Strong("Data type: "), html.Span(dtype)]
                        ),
                        dbc.ListGroupItem(
                            [html.Strong("Sample data: "), html.Span(str(sample_data))]
                        ),
                    ],
                    className="mb-3",
                ),
            ]
        )

    # Add collapsible card for the raw output
    unique_id = hash(f"{file_type}{str(hash(str(file_info)))}")
    components.append(
        html.Div(
            [
                dbc.Button(
                    "Show Raw Data",
                    id={"type": "collapse-button", "index": unique_id},
                    className="mb-3",
                    color="secondary",
                    size="sm",
                ),
                dbc.Collapse(
                    dbc.Card(
                        dbc.CardBody(
                            html.Pre(
                                json.dumps(file_info, indent=2),
                                style={"maxHeight": "300px", "overflowY": "auto"},
                            )
                        )
                    ),
                    id={"type": "collapse", "index": unique_id},
                    is_open=False,
                ),
            ]
        )
    )

    return html.Div(components)


@callback(
    Output({"type": "collapse", "index": dash.dependencies.MATCH}, "is_open"),
    [Input({"type": "collapse-button", "index": dash.dependencies.MATCH}, "n_clicks")],
    [State({"type": "collapse", "index": dash.dependencies.MATCH}, "is_open")],
)
def toggle_collapse(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open


@callback(
    Output({"type": "object-collapse", "index": dash.dependencies.MATCH}, "is_open"),
    [Input({"type": "object-button", "index": dash.dependencies.MATCH}, "n_clicks")],
    [State({"type": "object-collapse", "index": dash.dependencies.MATCH}, "is_open")],
)
def toggle_object_collapse(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open


@callback(
    Output("data-output", "children"),
    Output("loading-output", "children"),
    Output("files-being-analyzed", "children"),
    Output("error-message", "children"),
    Output("data-store", "data"),
    Input("upload-data", "contents"),
    Input("upload-data", "filename"),
    State("data-store", "data"),
)
def update_data_output(data_contents, data_filenames, stored_data_results):
    """
    Callback for data file upload.
    Files are automatically analyzed upon upload.

    Returns:
        - Data output component
        - Loading output
        - Files being analyzed text
        - Error message
        - Stored data results
    """
    if data_contents is None or data_filenames is None:
        # No file uploaded yet
        return html.Div(), "", "", "", {}

    # Initialize data store if not exists
    if stored_data_results is None:
        stored_data_results = {}

    # Check if the callback was triggered by a file upload
    ctx = dash.callback_context
    if not ctx.triggered or ctx.triggered[0]["prop_id"] == ".":
        return html.Div(), "", "", "", stored_data_results

    # Get the trigger
    changed_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if changed_id != "upload-data":
        return html.Div(), "", "", "", stored_data_results

    # Show the files being analyzed
    files_display = dbc.Alert(
        [
            html.H5("Analyzing Data Files:", className="alert-heading"),
            html.Ul([html.Li(filename) for filename in data_filenames]),
        ],
        color="info",
    )

    # Process each uploaded file
    output_components = []
    new_data_results = {}

    for content, filename in zip(data_contents, data_filenames):
        if content is None:
            continue

        # Save file to temporary directory
        try:
            file_path = save_uploaded_files(content, filename, "data")

            # Analyze file
            # All pickle files are analyzed securely by default
            file_ext = os.path.splitext(filename)[1].lower()
            result = analyze_data_file(file_path)

            if "error" in result:
                output_components.append(
                    dbc.Alert(
                        [
                            html.H5(f"Error analyzing {filename}"),
                            html.P(result["error"]),
                        ],
                        color="danger",
                        className="mb-3",
                    )
                )
                log.error(
                    "Error analyzing data file",
                    error=result["error"],
                    filename=filename,
                )
            else:
                # Store analysis results
                new_data_results[filename] = result

                # Format analysis results for display
                file_card = create_file_summary_card(filename, result)
                output_components.append(file_card)

        except Exception as e:
            output_components.append(
                dbc.Alert(
                    [
                        html.H5(f"Error processing {filename}"),
                        html.P(str(e)),
                    ],
                    color="danger",
                    className="mb-3",
                )
            )
            log.error(
                "Exception while processing data file", error=str(e), filename=filename
            )

    if not output_components:
        return html.Div("No valid data files uploaded."), "", "", "", {}

    # Wrap all cards in a row with responsive columns
    wrapped_output = dbc.Row(
        [dbc.Col(component, md=6, lg=4) for component in output_components]
    )

    return wrapped_output, "", "", "", new_data_results


@callback(
    Output("context-output", "children"),
    Output("context-store", "data"),
    Input("upload-context", "contents"),
    Input("upload-context", "filename"),
    State("context-store", "data"),
)
def update_context_output(context_contents, context_filenames, stored_context_results):
    """
    Callback for context file upload.
    Returns:
        - Context output component
        - Stored context results
    """
    if context_contents is None or context_filenames is None:
        # No file uploaded yet
        return html.Div("Upload context files to see extracted text."), {}

    # Initialize data store if not exists
    if stored_context_results is None:
        stored_context_results = {}

    # Check if the callback was triggered by a file upload
    ctx = dash.callback_context
    if not ctx.triggered or ctx.triggered[0]["prop_id"] == ".":
        return html.Div(
            "Upload context files to see extracted text."
        ), stored_context_results

    # Get the uploaded file(s)
    changed_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if changed_id != "upload-context":
        return html.Div(
            "Upload context files to see extracted text."
        ), stored_context_results

    # Process the uploaded files
    output_components = []
    new_context_results = stored_context_results.copy()

    for content, filename in zip(context_contents, context_filenames):
        if content is None:
            continue

        # Save file to temporary directory
        try:
            file_path = save_uploaded_files(content, filename, "context")

            # Extract text from file
            result = extract_text_from_file(file_path)

            if "error" in result:
                output_components.append(
                    dbc.Alert(
                        [
                            html.H5(f"Error processing {filename}"),
                            html.P(result["error"]),
                        ],
                        color="danger",
                        className="mb-3",
                    )
                )
                log.error(
                    "Error extracting text from file",
                    error=result["error"],
                    filename=filename,
                )
            else:
                # Store text content
                new_context_results[filename] = result.get("Text", "")

                # Display extracted text
                output_components.append(
                    dbc.Card(
                        [
                            dbc.CardHeader(html.H5(filename, className="mb-0")),
                            dbc.CardBody(
                                [
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    html.Strong("File Type: "),
                                                    html.Span(
                                                        result.get("Type", "Unknown")
                                                    ),
                                                ],
                                                md=6,
                                            ),
                                            dbc.Col(
                                                [
                                                    html.Strong("File Size: "),
                                                    html.Span(
                                                        result.get("Size", "Unknown")
                                                    ),
                                                ],
                                                md=6,
                                            ),
                                        ]
                                    ),
                                    html.Hr(),
                                    html.H6("Extracted Text:"),
                                    dbc.Card(
                                        dbc.CardBody(
                                            html.Pre(
                                                result.get("Text", "No text extracted"),
                                                style={
                                                    "whiteSpace": "pre-wrap",
                                                    "maxHeight": "300px",
                                                    "overflowY": "auto",
                                                },
                                            )
                                        ),
                                        className="bg-light mt-2",
                                    ),
                                ]
                            ),
                        ],
                        className="mb-3",
                    )
                )
        except Exception as e:
            output_components.append(
                dbc.Alert(
                    [
                        html.H5(f"Error processing {filename}"),
                        html.P(str(e)),
                    ],
                    color="danger",
                    className="mb-3",
                )
            )
            log.error(
                "Exception while processing context file",
                error=str(e),
                filename=filename,
            )

    if not output_components:
        return html.Div("No valid context files uploaded."), new_context_results

    # Wrap all cards in a container
    wrapped_output = html.Div(
        [
            html.H3("Context Files", className="mb-3"),
            dbc.Row([dbc.Col(component, md=6) for component in output_components]),
        ]
    )

    return wrapped_output, new_context_results


@callback(
    Output("correlation-output", "children"),
    Output("correlation-store", "data"),
    Input("analyze-correlations-button", "n_clicks"),
    State("data-store", "data"),
)
def update_correlation_analysis(n_clicks, data_results):
    """
    Callback for updating the correlation analysis.
    Returns:
        - Correlation output component
        - Correlation results data
    """
    # Check if callback was triggered
    if n_clicks is None or n_clicks == 0 or not data_results:
        return html.Div(
            "Click 'Analyze Correlations' to find connections between your data files."
        ), {}

    # Check if we have data results
    if not data_results or len(data_results) < 2:
        return dbc.Alert(
            "Upload at least two data files to analyze correlations.", color="warning"
        ), {}

    try:
        # Find correlations
        correlation_results = correlate_data_files(data_results)

        if "error" in correlation_results:
            return dbc.Alert(correlation_results["error"], color="warning"), {}

        # Create output components for correlations
        output_components = []

        # Shared columns
        shared_cols = correlation_results.get("shared_columns", {})
        if shared_cols:
            shared_cols_items = []
            for col, files in shared_cols.items():
                shared_cols_items.append(
                    dbc.ListGroupItem(
                        [html.Strong(col), ": Found in ", html.Span(", ".join(files))]
                    )
                )

            output_components.append(
                dbc.Card(
                    [
                        dbc.CardHeader(html.H5("Shared Columns", className="mb-0")),
                        dbc.CardBody(
                            [dbc.ListGroup(shared_cols_items, className="mb-3")]
                        ),
                    ],
                    className="mb-4",
                )
            )
        else:
            output_components.append(
                dbc.Alert("No shared columns found between files.", color="info")
            )

        # Potential join keys
        joins = correlation_results.get("potential_joins", [])
        if joins:
            joins_items = []
            for join in joins:
                joins_items.append(
                    dbc.ListGroupItem(
                        [
                            html.Strong(join["column"]),
                            ": Could join ",
                            html.Span(", ".join(join["files"])),
                        ]
                    )
                )

            output_components.append(
                dbc.Card(
                    [
                        dbc.CardHeader(
                            html.H5("Potential Join Keys", className="mb-0")
                        ),
                        dbc.CardBody([dbc.ListGroup(joins_items, className="mb-3")]),
                    ],
                    className="mb-4",
                )
            )

        # Similar sized files
        similar_sizes = correlation_results.get("similar_sizes", [])
        if similar_sizes:
            size_items = []
            for size_info in similar_sizes:
                size_items.append(
                    dbc.ListGroupItem(
                        [
                            html.Span(", ".join(size_info["files"])),
                            ": Row counts of ",
                            html.Span(
                                ", ".join(
                                    [str(count) for count in size_info["row_counts"]]
                                )
                            ),
                        ]
                    )
                )

            output_components.append(
                dbc.Card(
                    [
                        dbc.CardHeader(
                            html.H5("Files with Similar Sizes", className="mb-0")
                        ),
                        dbc.CardBody([dbc.ListGroup(size_items, className="mb-3")]),
                    ],
                    className="mb-4",
                )
            )

        # Common data types
        common_types = correlation_results.get("common_data_types", {})
        if common_types:
            types_components = []

            for files_key, types in common_types.items():
                type_items = []
                for col, dtype in types.items():
                    type_items.append(
                        dbc.ListGroupItem([html.Strong(col), ": ", html.Span(dtype)])
                    )

                types_components.append(
                    dbc.Card(
                        [
                            dbc.CardHeader(html.H6(files_key, className="mb-0")),
                            dbc.CardBody([dbc.ListGroup(type_items)]),
                        ],
                        className="mb-3",
                    )
                )

            output_components.append(
                dbc.Card(
                    [
                        dbc.CardHeader(html.H5("Common Data Types", className="mb-0")),
                        dbc.CardBody([html.Div(types_components)]),
                    ],
                    className="mb-4",
                )
            )

        return html.Div(output_components), correlation_results

    except Exception as e:
        log.error("Error in correlation analysis", error=str(e))
        return dbc.Alert(
            [
                html.H5("Error in Correlation Analysis"),
                html.P(str(e)),
                html.Pre(traceback.format_exc()),
            ],
            color="danger",
        ), {}


@callback(
    Output("ai-plan-output", "children"),
    Output("ai-plan-store", "data"),
    Input("generate-plan-button", "n_clicks"),
    State("data-store", "data"),
    State("context-store", "data"),
    State("correlation-store", "data"),
)
def update_analysis_plan(n_clicks, data_results, context_results, correlation_results):
    """
    Callback for generating AI analysis plan.
    Returns:
        - AI plan output component
        - AI plan data
    """
    if n_clicks is None or n_clicks == 0 or not data_results:
        return html.Div(
            "Click 'Generate Analysis Plan' to create an AI-powered plan."
        ), {}

    # Check if we have data results
    if not data_results:
        return dbc.Alert(
            "Upload data files first before generating an analysis plan.",
            color="warning",
        ), {}

    try:
        # Generate analysis plan
        plan_results = generate_analysis_plan(
            data_results, context_results or {}, correlation_results or {}
        )

        if "error" in plan_results:
            return dbc.Alert(
                [
                    html.H5("Error Generating Analysis Plan"),
                    html.P(plan_results["error"]),
                    html.Pre(plan_results.get("traceback", "")),
                ],
                color="danger",
            ), {}

        # Create output components for the plan
        output_components = []

        # Add timestamp
        if "timestamp" in plan_results:
            output_components.append(
                dbc.Alert(
                    [
                        html.Strong("Generated at: "),
                        html.Span(plan_results["timestamp"]),
                    ],
                    color="info",
                    className="mb-3",
                )
            )

        # Analysis plan
        if "analysis_plan" in plan_results:
            # Handle different formats of analysis_plan (string or list)
            if isinstance(plan_results["analysis_plan"], str):
                plan_text = plan_results["analysis_plan"]
                # Try to split on numbered items
                plan_items = re.split(r"\n\d+\.|\n\-", plan_text)
                if len(plan_items) > 1:
                    plan_items = [item.strip() for item in plan_items if item.strip()]
                    plan_content = html.Ol([html.Li(item) for item in plan_items])
                else:
                    plan_content = html.Pre(plan_text)
            elif isinstance(plan_results["analysis_plan"], list):
                plan_content = html.Ol(
                    [html.Li(item) for item in plan_results["analysis_plan"]]
                )
            else:
                plan_content = html.Pre(str(plan_results["analysis_plan"]))

            output_components.append(
                dbc.Card(
                    [
                        dbc.CardHeader(html.H5("Analysis Plan", className="mb-0")),
                        dbc.CardBody([plan_content]),
                    ],
                    className="mb-4",
                )
            )

        # Analysis code
        if "analysis_code" in plan_results:
            output_components.append(
                dbc.Card(
                    [
                        dbc.CardHeader(
                            html.H5("Generated Python Code", className="mb-0")
                        ),
                        dbc.CardBody(
                            [
                                html.Button(
                                    "Execute in E2B Sandbox",
                                    id="execute-code-button",
                                    className="btn btn-success mb-3",
                                ),
                                html.Pre(
                                    plan_results["analysis_code"],
                                    style={
                                        "backgroundColor": "#f8f9fa",
                                        "padding": "15px",
                                        "borderRadius": "5px",
                                        "maxHeight": "500px",
                                        "overflowY": "auto",
                                    },
                                ),
                            ]
                        ),
                    ],
                    className="mb-4",
                )
            )

        return html.Div(output_components), plan_results

    except Exception as e:
        log.error("Error generating analysis plan", error=str(e))
        return dbc.Alert(
            [
                html.H5("Error Generating Analysis Plan"),
                html.P(str(e)),
                html.Pre(traceback.format_exc()),
            ],
            color="danger",
        ), {}


@callback(
    Output("ai-results-output", "children"),
    Input("execute-code-button", "n_clicks"),
    State("ai-plan-store", "data"),
    State("data-store", "data"),
)
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
                df = pd.read_csv(file_path)
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

    # Sample data
    result["Sample Data"] = df.head(5).to_dict(orient="records")

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
            stats["unique_count"] = df[col].nunique()
            stats["top_values"] = {str(k): int(v) for k, v in value_counts.items()}

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
        "Path": file_path,
        "Name": file_name,
        "Size": f"{file_size / 1024:.2f} KB"
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
            result["Type"] = "Text File"

        elif ext == ".pdf":
            try:
                import PyPDF2

                with open(file_path, "rb") as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    for page in pdf_reader.pages:
                        text_content += page.extract_text() + "\n\n"
                result["Type"] = "PDF File"
            except ImportError:
                result["error"] = "PyPDF2 library not installed. Unable to process PDF."
                log.error("PyPDF2 library not installed")

        elif ext in [".docx", ".doc"]:
            try:
                import docx

                doc = docx.Document(file_path)
                for para in doc.paragraphs:
                    text_content += para.text + "\n"
                result["Type"] = "Word Document"
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
    if not data_results or len(data_results) < 2:
        return {"error": "Need at least two data files to find correlations"}

    correlations: Dict[str, Any] = {
        "shared_columns": {},
        "common_data_types": {},
        "similar_sizes": [],
        "potential_joins": [],
    }

    # Find common columns across DataFrames
    dataframes = {
        fname: info
        for fname, info in data_results.items()
        if info.get("Type") == "DataFrame"
    }

    # Exit early if we don't have at least 2 DataFrames
    if len(dataframes) < 2:
        return {
            "message": "Not enough DataFrame files found for correlation analysis",
            "dataframes_found": len(dataframes),
        }

    # Track all columns seen
    all_columns: Dict[str, List[str]] = {}

    # Find shared columns
    for fname, info in dataframes.items():
        columns = info.get("Columns", [])
        for col in columns:
            col_str = str(col)
            if col_str not in all_columns:
                all_columns[col_str] = []
            all_columns[col_str].append(fname)

    # Filter for columns that appear in multiple files
    correlations["shared_columns"] = {
        col: files for col, files in all_columns.items() if len(files) > 1
    }

    # Find potential join keys
    potential_joins: List[Dict[str, Any]] = []
    for col, files in correlations["shared_columns"].items():
        if len(files) >= 2:
            potential_joins.append({"column": col, "files": files})
    correlations["potential_joins"] = potential_joins

    # Find similar data types
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

    prompt = f"""
As a data science assistant, analyze the following datasets and context information to create an analysis plan.

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

The code should be self-contained and handle reading the files from their paths.
"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo",
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
