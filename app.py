import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from model_pipeline import load_model

app = FastAPI(title="House price API")
model = load_model("model.joblib")


class HouseFeatures(BaseModel):
    features: dict


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(body: HouseFeatures):
    df = pd.DataFrame([body.features])
    return {"price_tnd": float(model.predict(df)[0])}
