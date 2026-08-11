r"""serve_teja.py - my custom Phase 5 project.

A FastAPI service that loads
my trained Titanic survival classifier
and exposes a /predict endpoint.

Author: Venkat Teja Nallamothu
Date: 2026-08

Process:
    - Load a saved model from artifacts/.
    - Accept a POST request with passenger attributes.
    - Return the predicted survival outcome.

Data Source:
    - artifacts/model_teja.joblib (trained in model_builder_teja.py)

Terminal commands to run this service from the root project folder:

uv run fastapi dev src/mlstudio/serve_teja.py      # development (auto-reload)
uv run fastapi run src/mlstudio/serve_teja.py      # production

- OR -

uv run uvicorn mlstudio.serve_teja:app --reload    # development (auto-reload)
uv run uvicorn mlstudio.serve_teja:app             # production

Then send a request - open a new terminal and run

If macOS or Linux, use \ line continuation characters:

    curl -X POST http://127.0.0.1:8000/predict \
         -H "Content-Type: application/json" \
         -d '{"pclass": 1, "sex": "female", "age": 29, "sibsp": 0, "parch": 0, "fare": 100}'

If Windows (PowerShell), use ` instead of \ for line continuation:

    curl -X POST http://127.0.0.1:8000/predict `
         -H "Content-Type: application/json" `
         -d '{"pclass": 1, "sex": "female", "age": 29, "sibsp": 0, "parch": 0, "fare": 100}'

My Phase 5 modification:
    Copied serve_case.py and adapted it for my Titanic survival model:
    a different artifact path, a different feature set, and one extra
    validation step - the 'sex' feature arrives as a string ("male" or
    "female") and must be checked against the same encoding used in
    model_builder_teja.py before it can be passed to the model.
"""

# === Section 1. IMPORTS ===

import logging
from pathlib import Path
from typing import Any, Final

from datafun_toolkit.logger import get_logger, log_header
from fastapi import FastAPI, HTTPException
import joblib  # for serializing and deserializing the model
from sklearn.ensemble import RandomForestClassifier

__all__ = ["app", "predict_from_features", "predict"]

# === Section 2. CONFIGURE LOGGER ===

LOG: logging.Logger = get_logger("M06", level="DEBUG")
log_header(LOG, "M06")

# === Section 3. CONSTANTS AND CONFIGURATION ===

# The path to the saved model artifact.
MODEL_PATH: Final[Path] = Path("artifacts") / "model_teja.joblib"

# The feature columns the model was trained on.
# These must match exactly what was used during training.
FEATURE_COLS: Final[list[str]] = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
]

# Must match the encoding used in model_builder_teja.py.
SEX_MAP: Final[dict[str, int]] = {"male": 0, "female": 1}

# === Section 4. LOAD THE MODEL ===

LOG.info(f"Loading model from: {MODEL_PATH}")

if not MODEL_PATH.exists():
    LOG.error(f"Model file not found: {MODEL_PATH}")
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. Run mlstudio.model_builder_teja first."
    )

MODEL = joblib.load(MODEL_PATH)
LOG.info("Model loaded successfully")

# === Section 5. CREATE THE APP ===

app = FastAPI(title="Titanic survival classifier")

# === Section 6. DEFINE THE PREDICT ENDPOINT ===


def predict_from_features(
    model: RandomForestClassifier, payload: dict[str, Any]
) -> dict[str, Any]:
    """Pure prediction function - testable outside the web framework."""
    try:
        row: dict[str, Any] = {c: payload[c] for c in FEATURE_COLS}
    except KeyError as exc:
        raise ValueError(f"Missing required feature: {exc}") from exc

    sex_raw = row["sex"]
    if sex_raw not in SEX_MAP:
        raise ValueError(
            f"Invalid feature value for 'sex': expected one of "
            f"{list(SEX_MAP)}, got {sex_raw!r}"
        )
    row["sex"] = SEX_MAP[sex_raw]

    try:
        features = [float(row[c]) for c in FEATURE_COLS]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid feature value: {exc}") from exc

    label: int = int(model.predict([features])[0])
    return {"prediction": label, "survived": bool(label)}


@app.post("/predict")
def predict(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return predict_from_features(MODEL, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
