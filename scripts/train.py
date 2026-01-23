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
DATA_PATH = "dataset/winequality-white.csv"  # ensure dataset folder exists
OUTPUT_DIR = "artifacts"
MODEL_PATH = os.path.join(OUTPUT_DIR, "model.pkl")
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics.json")

TEST_SIZE = 0.5
RANDOM_STATE = 42
ALPHA = 1.0
# ----------------------------------------


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH, sep=";")

    X = df.drop("quality", axis=1)
    y = df["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=ALPHA))
    ])

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mse = float(mean_squared_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    # save model
    joblib.dump(model, MODEL_PATH)

    # save metrics
    metrics = {"mse": mse, "r2": r2}
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    print("===== Training Completed =====")
    print(f"MSE: {mse}")
    print(f"R2: {r2}")
    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
