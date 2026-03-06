import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "models/distilbert/best"

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()

dummy = tokenizer(
    "test input",
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=128,
)

torch.onnx.export(
    model,
    (dummy["input_ids"], dummy["attention_mask"]),
    "distilbert.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch"},
        "attention_mask": {0: "batch"},
    },
    opset_version=14,
)

print("ONNX export complete")