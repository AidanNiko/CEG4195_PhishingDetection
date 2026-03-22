import pandas as pd
import numpy as np
from pathlib import Path
from joblib import dump
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import mlflow
import mlflow.xgboost
import mlflow.sklearn

try:
    import lightgbm as lgb
    import mlflow.lightgbm

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not found — skipping. Install with: pip install lightgbm")


def _evaluate(model, X_test, y_test):
    """Compute standard classification metrics for any fitted sklearn-compatible model."""
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)
    binary_scores = proba[:, 1] if proba.shape[1] == 2 else None
    return {
        "preds": preds,
        "binary_scores": binary_scores,
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "roc_auc": (
            float(roc_auc_score(y_test, binary_scores))
            if binary_scores is not None
            else None
        ),
        "pr_auc": (
            float(average_precision_score(y_test, binary_scores))
            if binary_scores is not None
            else None
        ),
    }


def get_feature_importance(model, feature_names):
    """Return a feature importance array that works across all model types."""
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    elif hasattr(model, "coef_"):
        # Logistic Regression: absolute coefficient magnitude as proxy for importance
        return np.abs(model.coef_[0])
    return np.ones(len(feature_names))


def log_model_mlflow(model, artifact_path):
    """Pick the correct MLflow flavour for each model type."""
    if isinstance(model, XGBClassifier):
        mlflow.xgboost.log_model(model, artifact_path=artifact_path)
    elif LIGHTGBM_AVAILABLE and isinstance(model, lgb.LGBMClassifier):
        mlflow.lightgbm.log_model(model, artifact_path=artifact_path)
    else:
        mlflow.sklearn.log_model(model, artifact_path=artifact_path)


def train_model(file_path: str, config: dict, run_dir: Path):
    # Load the combined preprocessed dataset from parquet
    df = pd.read_parquet(file_path)

    training_cfg = config["training"]
    label_column = training_cfg.get("label_column", "label")
    drop_columns = training_cfg.get("drop_columns", ["body_text"])
    random_seed = training_cfg.get("random_seed", 42)
    test_size = training_cfg.get("test_size", 0.2)
    n_splits = training_cfg.get("cv_folds", 5)

    if label_column not in df.columns:
        raise ValueError(f"Missing required label column: {label_column}")

    # Drop raw text columns — the model uses extracted features, not raw text
    existing_drop_columns = [c for c in drop_columns if c in df.columns]
    if existing_drop_columns:
        df = df.drop(columns=existing_drop_columns)

    X = df.drop(label_column, axis=1)  # Feature matrix
    y = df[label_column]  # Target labels (0=legit, 1=phishing)

    # Stratified split preserves class ratio in both train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )

    # Build the list of (name, model_instance, params_dict) to train
    xgb_params = dict(training_cfg.get("xgboost", {}))
    xgb_params.setdefault("random_state", random_seed)

    rf_params = dict(training_cfg.get("random_forest", {}))
    rf_params.setdefault("random_state", random_seed)

    lr_params = dict(training_cfg.get("logistic_regression", {}))
    lr_params.setdefault("random_state", random_seed)

    models_to_train = [
        ("xgboost", XGBClassifier(**xgb_params), xgb_params),
        ("random_forest", RandomForestClassifier(**rf_params), rf_params),
        ("logistic_regression", LogisticRegression(**lr_params), lr_params),
    ]

    if LIGHTGBM_AVAILABLE:
        lgb_params = dict(training_cfg.get("lightgbm", {}))
        lgb_params.setdefault("random_state", random_seed)
        models_to_train.append(
            ("lightgbm", lgb.LGBMClassifier(**lgb_params), lgb_params)
        )

    all_results = []

    for model_name, model, params in models_to_train:
        print(f"\n--- Training {model_name} ---")
        model.fit(X_train, y_train)
        ev = _evaluate(model, X_test, y_test)

        print(f"Accuracy : {ev['accuracy']:.4f}")
        print(classification_report(y_test, ev["preds"]))

        # K-fold cross-validation on the full dataset for more robust estimates
        print(f"Running {n_splits}-fold CV for {model_name}...")
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        cv_results = cross_validate(
            model, X, y, cv=cv,
            scoring=["accuracy", "f1", "roc_auc", "precision", "recall"],
            n_jobs=-1,
        )
        cv_metrics = {
            "cv_accuracy_mean":  float(cv_results["test_accuracy"].mean()),
            "cv_accuracy_std":   float(cv_results["test_accuracy"].std()),
            "cv_f1_mean":        float(cv_results["test_f1"].mean()),
            "cv_f1_std":         float(cv_results["test_f1"].std()),
            "cv_roc_auc_mean":   float(cv_results["test_roc_auc"].mean()),
            "cv_roc_auc_std":    float(cv_results["test_roc_auc"].std()),
            "cv_precision_mean": float(cv_results["test_precision"].mean()),
            "cv_recall_mean":    float(cv_results["test_recall"].mean()),
        }
        print(f"CV F1:      {cv_metrics['cv_f1_mean']:.4f} ± {cv_metrics['cv_f1_std']:.4f}")
        print(f"CV ROC-AUC: {cv_metrics['cv_roc_auc_mean']:.4f} ± {cv_metrics['cv_roc_auc_std']:.4f}")

        metrics = {
            "accuracy": ev["accuracy"],
            "precision": ev["precision"],
            "recall": ev["recall"],
            "f1": ev["f1"],
            "roc_auc": ev["roc_auc"],
            "pr_auc": ev["pr_auc"],
            "rows": int(len(df)),
            "features": int(X.shape[1]),
            "test_size": float(test_size),
            "random_seed": int(random_seed),
        }

        # Save model file under run_dir/<model_name>/ — used by api.py to load the model
        model_dir = run_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{model_name}_model.joblib"
        dump(model, model_path)

        # one run per model — shows up side-by-side in the UI
        if mlflow.active_run():
            with mlflow.start_run(run_name=model_name, nested=True):
                mlflow.log_params({f"{model_name}_{k}": v for k, v in params.items()})
                mlflow.log_param("test_size", test_size)
                mlflow.log_param("random_seed", random_seed)
                mlflow.log_param("dataset_rows", metrics["rows"])
                mlflow.log_param("dataset_features", metrics["features"])
                mlflow.log_param("cv_folds", n_splits)
                mlflow.log_metrics(
                    {k: v for k, v in metrics.items() if isinstance(v, float)}
                )
                mlflow.log_metrics(cv_metrics)
                log_model_mlflow(model, artifact_path="model")

        print(f"Saved model to {model_path}")

        all_results.append(
            {
                "model_name": model_name,
                "model": model,
                "y_test": y_test,
                "y_pred": ev["preds"],
                "y_score": ev["binary_scores"],
                "feature_names": X_train.columns,
                "feature_importance": get_feature_importance(model, X_train.columns),
                "metrics": metrics,
                "model_path": model_path,
            }
        )

    best = max(all_results, key=lambda r: r["metrics"]["f1"])
    print(f"Best model by F1: {best['model_name']}  ({best['metrics']['f1']:.4f})")

    return all_results
