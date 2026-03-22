"""
FastAPI backend – POST /predict

Start with:
    python -m uvicorn api:app --reload

POST /predict
    Body (JSON): { "body": "...", "sender": "...", "receiver": "..." }
    Response:    { "prediction": 0|1, "label": "legitimate"|"phishing",
                   "confidence": 0.99, "phishing_probability": 0.01 }
"""

from pathlib import Path

import mlflow
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from Model.Predict import predict_email
from pipeline_config import CONFIG

app = FastAPI(title="Phishing Email Detector", version="1.0.0")


@app.get("/")
def frontend():
    """Serve the HTML frontend."""
    return FileResponse(Path(__file__).parent / "frontend.html")


# Allow the local HTML frontend to call the API from a browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)


class EmailRequest(BaseModel):
    body: str
    sender: str
    receiver: str


def _get_latest_run_dir() -> Path:
    """Scan the runs directory and return the most recently created run folder."""
    runs_root = Path(CONFIG["paths"].get("runs_dir", "Docs"))
    # Folders are named run_YYYYMMDD_HHMMSS so lexicographic sort = chronological sort
    runs = sorted(runs_root.glob("run_*"), key=lambda p: p.name)
    if not runs:
        raise RuntimeError(
            f"No training runs found under '{runs_root}'. Run the pipeline first."
        )
    return runs[-1]


# Load the model once at server startup rather than on every request.
try:
    _RUN_DIR = _get_latest_run_dir()
    # Warm the embedding model so the first API call isn't slow
    from Preprocessing.Helper_Functions import get_embed_model

    get_embed_model()
    print(f"Using run: {_RUN_DIR}")
except Exception as _e:
    _RUN_DIR = None
    print(f"Warning: could not load model at startup – {_e}")


@app.post("/predict")
def predict(request: EmailRequest, model: str = "lightgbm"):
    """Classify a single email as legitimate (0) or phishing (1)."""
    if _RUN_DIR is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available. Run the training pipeline first.",
        )
    try:
        result = predict_email(
            body=request.body,
            sender=request.sender,
            receiver=request.receiver,
            run_dir=_RUN_DIR,
            model_name=model,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    # Apply confidence threshold — scores in the uncertain zone are not committed
    # to either class, giving the user a clearer signal about borderline emails.
    threshold = CONFIG.get("phishing_threshold", 0.7)
    prob = result["phishing_probability"]
    if prob >= threshold:
        result["label"] = "phishing"
        result["prediction"] = 1
    elif prob <= (1 - threshold):
        result["label"] = "legitimate"
        result["prediction"] = 0
    else:
        result["label"] = "uncertain"
        result["prediction"] = -1

    # Log this prediction to MLflow
    try:
        mlflow.set_experiment("frontend_predictions")
        with mlflow.start_run(run_name=f"{model}_{result['label']}"):
            mlflow.set_tag("model", model)
            mlflow.set_tag("sender", request.sender)
            mlflow.set_tag("receiver", request.receiver)
            mlflow.log_metric("phishing_probability", result["phishing_probability"])
            mlflow.log_metric("confidence", result["confidence"])
            mlflow.log_param("label", result["label"])
            mlflow.log_param("body_length", len(request.body))
    except Exception:
        pass  # Never let MLflow logging break the API response

    return result


@app.get("/health")
def health():
    return {
        "status": "ok",
        "run_dir": str(_RUN_DIR) if _RUN_DIR else None,
    }
