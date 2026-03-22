# CEG4195 – Phishing Email Detection

Machine learning system that classifies emails as **phishing** or **legitimate** using LightGBM, Random Forest, XGBoost, and Logistic Regression models trained on the Enron, Nazario, and SpamAssassin/EmailOrigin datasets.

---

## Running with Docker (Recommended)

**Prerequisite:** [Docker Desktop](https://www.docker.com/products/docker-desktop) installed and running.

### 1. Download the project files

Either clone the GitHub repository:
```
git clone https://github.com/AidanNiko/CEG4195_PhishingDetection.git
cd CEG4195_PhishingDetection
```
Or unzip the submitted archive.

### 2. Start the application

```
docker compose up
```

This will automatically pull the pre-built image from Docker Hub — no Python installation required. The first run downloads the image (~3 GB) so may take a few minutes.

### 3. Open the frontend

- **Phishing Detector UI** → http://localhost:8001
- **MLflow Experiment Tracker** → http://localhost:5000

### 4. Stop the application

```
docker compose down
```

---

## Running without Docker

**Prerequisites:** Python 3.11 or 3.12, pip

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Start the API

```
python -m uvicorn api:app --port 8001
```

### 3. Open the frontend

Open `frontend.html` directly in a browser, or navigate to http://localhost:8001

### 4. (Optional) View MLflow results

```
python -m mlflow ui
```
Then open http://localhost:5000

---

## Running the Training Pipeline

To retrain all models from scratch (requires the raw dataset files in `Datasets/`):

```
python main_pipeline.py
```

Trained models are saved under `Docs/run_<timestamp>/` and MLflow metrics are logged to `mlruns/`.

---

## Project Structure

```
api.py                  – FastAPI prediction endpoint (port 8001)
frontend.html           – Web UI
main_pipeline.py        – Full training pipeline
pipeline_config.py      – Hyperparameters and path configuration
requirements.txt        – Python dependencies
Dockerfile              – Container build definition
docker-compose.yml      – Runs API + MLflow UI together
Model/
  Train_Models.py       – Model training with k-fold cross-validation
  Predict.py            – Inference logic
Preprocessing/
  Enron_Clean.py        – Enron corpus preprocessing
  Nazario_Clean.py      – Nazario phishing corpus preprocessing
  EmailOrigin_Clean.py  – SpamAssassin/EmailOrigin preprocessing
  Helper_Functions.py   – Sentence-BERT feature extraction
Docs/                   – Saved model files (.joblib) from last training run
```
