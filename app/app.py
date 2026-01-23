from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Model path inside docker container
MODEL_PATH = "model/model.pkl"
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Wine Quality Inference API", version="1.0")


class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float


@app.get("/")
def home():
    return {"message": "Wine Quality Inference API running"}


@app.post("/predict")
def predict(features: WineFeatures):
    input_data = np.array([[
        features.fixed_acidity,
        features.volatile_acidity,
        features.citric_acid,
        features.residual_sugar,
        features.chlorides,
        features.free_sulfur_dioxide,
        features.total_sulfur_dioxide,
        features.density,
        features.pH,
        features.sulphates,
        features.alcohol
    ]])

    pred = model.predict(input_data)[0]

    # Required output format for Lab 4
    return {
        "name": "Chemma Venkata Yaswanth",
        "roll_no": "2022BCS0049",
        "wine_quality": int(round(float(pred)))
    }
