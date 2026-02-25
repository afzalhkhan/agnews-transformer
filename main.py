import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import time

MODEL_PATH = "models/distilbert-imdb/best"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()


def predict(text: str):
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
        probs = F.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][predicted_class].item()

    latency_ms = (time.time() - start) * 1000

    label = "positive" if predicted_class == 1 else "negative"

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "latency_ms": round(latency_ms, 2),
    }


if __name__ == "__main__":
    while True:
        text = input("\nEnter text (or 'quit'): ")
        if text.lower() == "quit":
            break
        result = predict(text)
        print(result)
