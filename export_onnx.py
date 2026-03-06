import torch
from transformers import AutoModelForSequenceClassification

MODEL_PATH = "models/distilbert-agnews/best"

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

dummy_input_ids = torch.randint(0, 1000, (1, 128))
dummy_attention_mask = torch.ones((1, 128))

torch.onnx.export(
    model,
    (dummy_input_ids, dummy_attention_mask),
    "models/distilbert.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch"},
        "attention_mask": {0: "batch"},
    },
    opset_version=13,
)

print("ONNX export complete")