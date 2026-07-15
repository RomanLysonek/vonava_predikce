"""Local presentation dashboard for the Notino forecast results.

Serves the static frontend (webapp/static/) and a small JSON API that reads
outputs/results.json fresh on every request -- rerun the ML pipeline
(uv run python ml/pipeline.py) or the lightweight
`uv run python ml/export_results.py`, then just refresh the browser.

Run (from repo root): uv run python webapp/server.py
Then open:            http://127.0.0.1:8999
"""

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
RESULTS_PATH = ROOT_DIR / "outputs" / "results.json"

app = FastAPI(title="Notino Forecast Dashboard")


@app.get("/api/results")
def get_results() -> JSONResponse:
    if not RESULTS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "outputs/results.json not found. Run "
                "'uv run python ml/pipeline.py' (or ml/export_results.py) first."
            ),
        )
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    return JSONResponse(data)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    """Serve the explicit SVG favicon for browsers that still request /favicon.ico."""
    return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")


@app.get("/dataset")
def dataset_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "dataset.html")


@app.get("/evaluation")
def evaluation_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "evaluation.html")


@app.get("/model/{slug}")
def model_page(slug: str) -> FileResponse:
    # One shared template; model.js reads `slug` from the URL itself and
    # renders that model's data/colors. Unknown slugs still get the page --
    # model.js shows a clear "not found" state rather than a 404.
    return FileResponse(STATIC_DIR / "model.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=8999, reload=True, reload_dirs=[str(Path(__file__).parent)])
