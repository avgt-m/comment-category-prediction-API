from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(
    title="Comment Category Prediction API",
    description="ML API for comment category classification.",
    version="1.0.0",
)

MODEL_PATH = "model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    model_loaded = True
    model_error = None
except Exception as e:
    model = None
    model_loaded = False
    model_error = str(e)


class PredictionRequest(BaseModel):
    comment: str

@app.get("/")
def root():
    return {
        "message": "Comment Category Prediction API",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
def health():
    return {
        "status": "ok" if model_loaded else "error",
        "model_loaded": model_loaded,
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    if not model_loaded:
        return {
            "error": "Model could not be loaded",
            "details": model_error,
        }

    prediction = model.predict([request.comment])[0]

    return {
        "prediction": int(prediction)
    }
