import json
import pandas as pd
from pathlib import Path
from joblib import load

from Preprocessing.Helper_Functions import (
    extract_urls_from_text,
    url_features,
    get_domain,
    domain_features,
    body_text_features,
    get_embed_model,
)

LABEL_NAMES = {0: "legitimate", 1: "phishing"}


def _extract_features(body: str, sender: str, receiver: str) -> dict:
    """Run the same feature extraction pipeline used during training on a single email."""
    # Extract domain from the email address (e.g. "user@gmail.com" -> "gmail.com")
    sender_domain = get_domain(sender)
    receiver_domain = get_domain(receiver)

    # Find any URLs in the body and extract features from the first one.
    urls = extract_urls_from_text(body)
    url_feat = (
        url_features(urls[0])
        if urls
        else {
            "url_length": 0,
            "url_num_dots": 0,
            "url_has_ip": False,
            "url_num_special_chars": 0,
            "url_uses_https": False,
        }
    )

    # Statistical body features: length, exclamations, questions, uppercase count, URL count
    body_feat = body_text_features(body)

    # Encode the body as a 384-dimensional semantic embedding via sentence-transformers.
    embed_model = get_embed_model()
    embedding = embed_model.encode([body], show_progress_bar=False)[0]
    embed_feats = {f"embed_{i}": float(v) for i, v in enumerate(embedding)}

    return {
        **url_feat,
        **body_feat,
        **domain_features(sender_domain, prefix="sender_"),
        **domain_features(receiver_domain, prefix="receiver_"),
        "domains_match": (sender_domain == receiver_domain and sender_domain != ""),
        **embed_feats,
    }


SUPPORTED_MODELS = ["xgboost", "random_forest", "logistic_regression", "lightgbm"]


def predict_email(body: str, sender: str, receiver: str, run_dir: Path, model_name: str = "xgboost") -> dict:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {SUPPORTED_MODELS}")
    model_path = run_dir / model_name / f"{model_name}_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = load(model_path)

    features = _extract_features(body, sender, receiver)

    # Wrap in a single-row DataFrame so XGBoost receives the expected input shape
    df = pd.DataFrame([features])
    pred = int(model.predict(df)[0])
    proba = model.predict_proba(df)[0]  # [prob_legit, prob_phishing]

    return {
        "prediction": pred,  # 0 = legitimate, 1 = phishing
        "label": LABEL_NAMES.get(pred, str(pred)),  # Human-readable label
        "confidence": float(proba[pred]),  # Probability of the predicted class
        "phishing_probability": float(proba[1]),  # Always the raw phishing score
    }
