import json
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# ---------------- CONFIG ----------------
DATA_PATH = "dataset/winequality-white.csv"

# Primary output (for CI/CD - Lab4)
ARTIFACTS_DIR = "artifacts"
MODEL_ARTIFACT_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
METRICS_ARTIFACT_PATH = os.path.join(ARTIFACTS_DIR, "metrics.json")

# Secondary output (for Docker - Lab6 & Lab7)
MODEL_DIR = "model"
MODEL_DEPLOY_PATH = os.path.join(MODEL_DIR, "model.pkl")
METRICS_DEPLOY_PATH = os.path.join(MODEL_DIR, "metrics.json")

TEST_SIZE = 0.4
RANDOM_STATE = 42
ALPHA = 1.0
# ----------------------------------------


def main():
    # Create required folders
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load dataset
    df = pd.read_csv(DATA_PATH, sep=";")

    X = df.drop("quality", axis=1)
    y = df["quality"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Model pipeline
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=ALPHA))
    ])

    # Train
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Metrics
    mse = float(mean_squared_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))
    metrics = {"mse": mse, "r2": r2}

    # -------- Save to artifacts (Lab4 CI/CD) --------
    joblib.dump(model, MODEL_ARTIFACT_PATH)
    with open(METRICS_ARTIFACT_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    # -------- Save to model folder (Docker deployment) --------
    joblib.dump(model, MODEL_DEPLOY_PATH)
    with open(METRICS_DEPLOY_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    print("===== Training Completed =====")
    print(f"MSE: {mse}")
    print(f"R2: {r2}")
    print(f"Saved -> {MODEL_ARTIFACT_PATH}")
    print(f"Saved -> {MODEL_DEPLOY_PATH}")


if __name__ == "__main__":
    main()