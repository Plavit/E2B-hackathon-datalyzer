# Run with `modal serve deploy.py` or deploy stably with `modal deploy deploy.py`
import modal

from e2b_hackathon.dash_app import app as web_app


image = modal.Image.debian_slim().pip_install(
    [
        "e2b-code-interpreter",
        "python-dotenv",
        "pandas",
        "dash",
        "dash-bootstrap-components",
        "plotly",
        "structlog",
        "pyarrow",
        "fastparquet",
        "python-docx",
        "pypdf2",
        "openai",
        "openpyxl",
        "matplotlib",
        "seaborn",
    ]
)
image_with_source = image.add_local_python_source("e2b_hackathon")
app = modal.App("d2ma", image=image_with_source)


@app.function(
    image=image,
    allow_concurrent_inputs=1000,
    secrets=[modal.Secret.from_name("e2b-hackathon")],
)
@modal.wsgi_app()
def flask_app():
    return web_app.server
