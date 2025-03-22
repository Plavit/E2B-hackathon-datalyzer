import os
import base64
import tempfile
import json
import traceback
from typing import List, Dict, Any

import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc  # type: ignore

# import plotly.graph_objects as go  # type: ignore
import plotly.express as px  # type: ignore
import structlog

from e2b_hackathon.pickle_analyzer import analyze_pickle_files

# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="Datalyzer",
)
log = structlog.get_logger()

# Define application layout
layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H1("Pickle File Analyzer", className="text-center my-4"),
                        html.P(
                            "Upload pickle files to analyze their content and structure using E2B sandbox",
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
                                        html.H5("Upload Files", className="card-title"),
                                        html.P(
                                            "Upload one or more pickle files (.pkl or .pickle) to analyze",
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
                                                    html.A("Select Pickle Files"),
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
                                    ]
                                )
                            ],
                            className="mb-4",
                        ),
                    ],
                    width=12,
                )
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
                        html.Div(id="output-data-upload", className="mt-4"),
                    ],
                    width=12,
                )
            ]
        ),
        # Footer
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Hr(),
                        html.P(
                            "Powered by E2B Sandbox - A secure environment for analyzing pickle files",
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


def save_uploaded_files(contents: List[str], filenames: List[str]) -> List[str]:
    """
    Save uploaded files to temporary directory and return paths

    Args:
        contents: List of file contents
        filenames: List of filenames

    Returns:
        List of paths to saved files
    """
    temp_dir = tempfile.mkdtemp()
    saved_files = []

    for content, filename in zip(contents, filenames):
        # Check if the file is a pickle file (basic check by extension)
        if not (filename.endswith(".pkl") or filename.endswith(".pickle")):
            continue

        # Decode the file content
        content_type, content_string = content.split(",")
        decoded = base64.b64decode(content_string)

        # Save to temporary file
        file_path = os.path.join(temp_dir, filename)
        with open(file_path, "wb") as f:
            f.write(decoded)

        saved_files.append(file_path)

    return saved_files


def parse_analysis_results(analysis_text: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse the raw analysis text to extract structured information

    Args:
        analysis_text: The raw text output from pickle_analyzer

    Returns:
        Dictionary with parsed analysis results
    """
    # Split into sections by file
    sections = analysis_text.split("=" * 50)
    sections = [s.strip() for s in sections if s.strip()]

    results = {}

    for section in sections:
        lines = section.split("\n")
        if len(lines) < 2:
            continue

        # Extract filename from the first line
        file_name = lines[0].replace("Analysis of ", "").replace(":", "")

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

    if file_type == "dict":
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
    [
        Output("output-data-upload", "children"),
        Output("loading-output", "children"),
        Output("files-being-analyzed", "children"),
        Output("error-message", "children"),
    ],
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True,
)
def update_output(contents, filenames):
    if contents is None:
        return html.Div(), "", "", ""

    # Show the files being analyzed
    files_display = (
        html.Div(
            [
                dbc.Alert(
                    [
                        html.H5("Analyzing Files:", className="alert-heading"),
                        html.Ul([html.Li(filename) for filename in filenames]),
                    ],
                    color="info",
                )
            ]
        )
        if filenames
        else ""
    )

    # Save uploaded files to disk
    try:
        saved_files = save_uploaded_files(contents, filenames)
        log.info("Files saved", count=len(saved_files), filenames=filenames)
    except Exception as e:
        log.error("Error saving files", error=str(e), exc_info=True)
        return (
            html.Div(),
            "",
            "",
            dbc.Alert(
                f"Error saving uploaded files: {str(e)}",
                color="danger",
            ),
        )

    if not saved_files:
        return (
            html.Div(),
            "",
            "",
            dbc.Alert(
                "No valid pickle files were uploaded. Please upload files with .pkl or .pickle extension.",
                color="warning",
            ),
        )

    try:
        # Create a string buffer to capture the printed output
        import io
        import sys

        # Redirect stdout to capture output
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout

        # Log file info before analysis
        for file_path in saved_files:
            file_size = os.path.getsize(file_path)
            log.info("Analyzing file", path=file_path, size=file_size)

        # Run the analysis
        analyze_pickle_files(saved_files, verbose=True)

        # Restore stdout
        sys.stdout = old_stdout
        analysis_output = new_stdout.getvalue()

        log.info("Analysis completed", output_length=len(analysis_output))

        # Parse results
        results = parse_analysis_results(analysis_output)

        # Create output elements
        output_elements = []

        # Add summary cards for each file
        output_elements.append(html.H3("Analysis Results", className="mt-4 mb-3"))

        if not results:
            output_elements.append(
                dbc.Alert(
                    "No results were obtained from the analysis. The files might be corrupted or not contain valid pickle data.",
                    color="warning",
                )
            )
        else:
            file_cards_row = []

            for file_name, file_info in results.items():
                # Create a card for each file
                if file_name.startswith("[2m"):
                    log.info("Skipping overall results", file_info=file_info)
                    continue
                card = create_file_summary_card(file_name, file_info)
                file_cards_row.append(dbc.Col(card, md=6, lg=4))

            # Add cards in a row with responsive columns
            output_elements.append(dbc.Row(file_cards_row))

            # Add collapsible raw output section
            output_elements.append(
                html.Div(
                    [
                        html.H4("Raw Analysis Output", className="mt-4"),
                        dbc.Button(
                            "Toggle Raw Output",
                            id="toggle-raw-output",
                            className="mb-3",
                            color="secondary",
                        ),
                        dbc.Collapse(
                            dbc.Card(
                                dbc.CardBody(
                                    html.Pre(
                                        analysis_output,
                                        style={
                                            "backgroundColor": "#f8f9fa",
                                            "padding": "15px",
                                            "borderRadius": "5px",
                                            "maxHeight": "500px",
                                            "overflowY": "auto",
                                        },
                                    )
                                )
                            ),
                            id="collapse-raw-output",
                            is_open=False,
                        ),
                    ]
                )
            )

        # Clean up temporary files
        for file_path in saved_files:
            try:
                os.remove(file_path)
            except Exception as e:
                log.error("Error removing file", path=file_path, error=str(e))

        # Try to remove the temp directory
        try:
            os.rmdir(os.path.dirname(saved_files[0]))
        except Exception as e:
            log.error("Error removing temp directory", error=str(e))

        return html.Div(output_elements), "", files_display, ""

    except Exception as e:
        error_msg = str(e)
        tb = traceback.format_exc()
        log.error("Pickle file upload error", error=error_msg, traceback=tb)

        # Return error message with detailed information
        return (
            html.Div(),
            "",
            "",
            dbc.Alert(
                [
                    html.H4("Error", className="alert-heading"),
                    html.P(f"An error occurred during analysis: {error_msg}"),
                    html.Hr(),
                    html.Details(
                        [
                            html.Summary("Technical Details (click to expand)"),
                            html.Pre(tb, style={"whiteSpace": "pre-wrap"}),
                        ]
                    ),
                    html.Hr(),
                    html.P(
                        "Please check that your pickle files are valid and try again.",
                        className="mb-0",
                    ),
                ],
                color="danger",
            ),
        )


@callback(
    Output("collapse-raw-output", "is_open"),
    [Input("toggle-raw-output", "n_clicks")],
    [State("collapse-raw-output", "is_open")],
)
def toggle_raw_output(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open


def main():
    """Run the Dash app server"""
    app.run(debug=True)


if __name__ == "__main__":
    main()
