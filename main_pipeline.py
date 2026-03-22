import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
import mlflow
from Model.Train_Models import train_model
from pipeline_config import CONFIG

# Preprocessing imports with aliases
from Preprocessing.Enron_Clean import clean_enron as clean_enron_metadata
from Preprocessing.Nazario_Clean import clean_nazario as clean_nazario_metadata
from Preprocessing.EmailOrigin_Clean import clean_email_origin


def main():
    config = CONFIG
    parquet_path = Path(config["paths"]["dataset_parquet"])
    sample_size = config["preprocessing"]["sample_size"]
    shuffle_seed = config["preprocessing"].get("shuffle_seed", 42)

    # Each run gets a timestamped directory under Docs/ so results are never overwritten
    run_root = Path(config["paths"].get("runs_dir", "Models/runs"))
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = run_root / run_id
    plots_dir = run_dir / "plots"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Configure MLflow experiment
    mlflow_cfg = config.get("mlflow", {})
    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "mlruns"))
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "phishing-detection"))

    # Wrap the entire pipeline in a single MLflow run so all artifacts, metrics, and parameters are grouped together in the UI
    with mlflow.start_run(run_name=run_id):
        mlflow.set_tag("run_id", run_id)
        mlflow.set_tag("dataset_parquet", str(parquet_path))

        # ---------------------------------------------------------------------------------
        # PREPROCESSING STEP
        # ---------------------------------------------------------------------------------

        # If a parquet already exists we skip preprocessing to save time.
        if parquet_path.exists():
            print(f"Found existing dataset at {parquet_path}. Skipping preprocessing.")
        else:
            # Clean and extract features from each data source independently
            enron_df = clean_enron_metadata(sample_size=sample_size)  # label=0 (legit)
            nazario_df = clean_nazario_metadata(
                sample_size=sample_size
            )  # label=1 (phishing)
            email_origin_sample = config["preprocessing"].get(
                "email_origin_sample_size", 18000
            )
            email_origin_df = clean_email_origin(
                sample_size=email_origin_sample, row_label=1
            )  # label=1 (spam)

            email_origin_ham_sample = config["preprocessing"].get(
                "email_origin_ham_sample_size", 10000
            )
            email_origin_ham_df = clean_email_origin(
                sample_size=email_origin_ham_sample, row_label=0
            )  # label=0 (legit, diverse domains)

            # Combine all sources into one shuffled dataset
            combined_df = pd.concat(
                [enron_df, nazario_df, email_origin_df, email_origin_ham_df],
                ignore_index=True,
            )
            combined_df = combined_df.sample(
                frac=1, random_state=shuffle_seed
            ).reset_index(drop=True)
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            # Save as parquet — much faster to read back than CSV for large datasets
            combined_df.to_parquet(parquet_path)
            print(f"Preprocessed dataset saved to {parquet_path}.")

        # ---------------------------------------------------------------------------------
        # TRAINING STEP
        # ---------------------------------------------------------------------------------
        print("Proceeding to training...")
        training_results = train_model(parquet_path, config=config, run_dir=run_dir)

        # ---------------------------------------------------------------------------------
        # EVALUATION PLOTS STEP — one set of plots per model
        # ---------------------------------------------------------------------------------
        show_plots = config.get("plots", {}).get("show_plots", False)
        top_n      = config.get("plots", {}).get("top_n_features", 20)

        for result in training_results:
            model_plots_dir = plots_dir / result["model_name"]
            plot_model_evaluation(
                result,
                output_dir=model_plots_dir,
                show_plots=show_plots,
                top_n=top_n,
            )

        # Log all saved plot images as artifacts so they appear in the MLflow UI
        if plots_dir.exists():
            mlflow.log_artifacts(str(plots_dir), artifact_path="plots")

        # Log model paths as tags so they're findable from the MLflow UI
        for r in training_results:
            mlflow.set_tag(f"{r['model_name']}_path", str(r["model_path"]))

    print(
        f"MLflow run '{run_id}' complete. Run `mlflow ui` in the project root to compare runs."
    )


def plot_model_evaluation(
    results, output_dir: Path, show_plots: bool = False, top_n: int = 20
):
    """Generate and save performance curves and feature importance for one model."""
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = results.get("model_name", "model")
    y_test  = results["y_test"]
    y_pred  = results["y_pred"]
    y_score = results.get("y_score", None)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Confusion Matrix — shows true vs predicted labels as a normalised heatmap
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        normalize="true",
        cmap="Blues",
        ax=axes[0],
        colorbar=False,
    )
    axes[0].set_title(f"Confusion Matrix — {model_name}")

    # ROC Curve — plots true positive rate vs false positive rate across thresholds
    if y_score is not None:
        RocCurveDisplay.from_predictions(
            y_test,
            y_score,
            ax=axes[1],
        )
        axes[1].set_title(f"ROC Curve — {model_name}")
    else:
        axes[1].axis("off")

    # Precision-Recall Curve — more informative than ROC for imbalanced classes
    if y_score is not None:
        PrecisionRecallDisplay.from_predictions(
            y_test,
            y_score,
            ax=axes[2],
        )
        axes[2].set_title(f"Precision-Recall Curve — {model_name}")
    else:
        axes[2].axis("off")

    plt.tight_layout()
    perf_plot = output_dir / "performance_curves.png"
    fig.savefig(perf_plot, dpi=200, bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    # Feature Importance — which features the model relied on most to split decisions
    importances = results["feature_importance"]
    features = results["feature_names"]

    # Take the top_n most important features by their raw importance score
    idx = np.argsort(importances)[-top_n:]

    fig_fi, ax = plt.subplots(figsize=(8, 6))
    ax.barh(np.array(features)[idx], importances[idx])
    ax.set_xlabel("Importance score")
    ax.set_ylabel("Features")
    ax.set_title(f"Top Feature Importance — {model_name}")

    for i, v in enumerate(importances[idx]):
        ax.text(v, i, f"{v:.4f}")

    plt.tight_layout()
    fi_plot = output_dir / "feature_importance.png"
    fig_fi.savefig(fi_plot, dpi=200, bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close(fig_fi)

    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
