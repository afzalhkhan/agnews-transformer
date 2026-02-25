from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time

MODEL_PATH = "models/distilbert-agnews/best"   # <-- new model path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()
print("Model loaded on", device)

# AG News label mapping
LABEL_MAP = {
    0: "world",
    1: "sports",
    2: "business",
    3: "sci_tech",
}


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    confidence: float
    latency_ms: float
    scores: Dict[str, float]


app = FastAPI(title="Transformer News Classification API")


def predict_text(text: str) -> PredictResponse:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    ).to(device)

    start = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)[0]

    latency_ms = (time.time() - start) * 1000

    predicted_class = int(torch.argmax(probs).item())
    confidence = float(probs[predicted_class].item())

    label = LABEL_MAP.get(predicted_class, str(predicted_class))

    scores = {
        LABEL_MAP[i]: float(probs[i].item())
        for i in range(len(LABEL_MAP))
    }

    return PredictResponse(
        label=label,
        confidence=round(confidence, 4),
        latency_ms=round(latency_ms, 2),
        scores={k: round(v, 4) for k, v in scores.items()},
    )


@app.get("/")
def root():
    return {"status": "ok", "message": "Transformer news classification API is running"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    return predict_text(req.text)