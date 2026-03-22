CONFIG = {
    "paths": {
        "dataset_parquet": "Datasets/email_dataset.parquet",
        "runs_dir": "Docs",
    },
    "preprocessing": {
        "sample_size": 10000,
        "email_origin_sample_size": 18000,
        "email_origin_ham_sample_size": 10000,
        "shuffle_seed": 42,
    },
    "training": {
        "test_size": 0.2,
        "random_seed": 42,
        "cv_folds": 5,
        "label_column": "label",
        "drop_columns": ["body_text"],
        "xgboost": {
            "n_estimators": 300,
            "max_depth": 8,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "logloss",
            "scale_pos_weight": 0.7,
        },
        "random_forest": {
            "n_estimators": 300,
            "max_depth": 20,
            "n_jobs": -1,
            "class_weight": "balanced",
        },
        "logistic_regression": {
            "max_iter": 1000,
            "solver": "lbfgs",
            "n_jobs": -1,
            "class_weight": "balanced",
        },
        "lightgbm": {
            "n_estimators": 300,
            "max_depth": 8,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": 0.7,
            "n_jobs": -1,
            "verbose": -1,
        },
    },
    "plots": {
        "show_plots": False,
        "top_n_features": 20,
    },
    # 0.7 means the model must be at least 70% confident before flagging.
    "phishing_threshold": 0.7,
    "mlflow": {
        # Run `mlflow ui` in the project root to browse results.
        "experiment_name": "phishing-detection",
        # Local directory where MLflow stores run data.
        "tracking_uri": "mlruns",
    },
}
