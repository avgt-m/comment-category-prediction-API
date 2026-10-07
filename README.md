# Comment Category Prediction API

A FastAPI deployment of a text classification model trained on the Comment Category Prediction dataset.

## Model

The model is a scikit-learn Pipeline containing:
- TF-IDF vectorization with word unigrams/bigrams
- Logistic Regression classifier

The model was trained on 20,000 examples for this deployment exercise.

## API

### GET /health

Returns whether the model loaded successfully.

Example response:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### POST /predict

Request body:

```json
{
  "comment": "This is an example comment."
}
```

Example response:

```json
{
  "prediction": 0
}
```

## Run locally

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Open http://127.0.0.1:8000/docs

## Docker

```bash
docker build -t comment-category-api .
docker run -p 8000:8000 comment-category-api
```
