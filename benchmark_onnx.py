import time
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 4
sess_options.inter_op_num_threads = 4

session = ort.InferenceSession(
    "distilbert.onnx",
    sess_options=sess_options,
    providers=["CPUExecutionProvider"]
)

TEXT = "NASA launches a new satellite to study climate change."

inputs = tokenizer(
    TEXT,
    return_tensors="np",
    padding="max_length",
    truncation=True,
    max_length=128,
)

onnx_inputs = {
    "input_ids": inputs["input_ids"].astype(np.int64),
    "attention_mask": inputs["attention_mask"].astype(np.int64),
}

# warmup
for _ in range(10):
    session.run(None, onnx_inputs)

times = []

for _ in range(100):

    start = time.time()

    session.run(None, onnx_inputs)

    elapsed = (time.time() - start) * 1000
    times.append(elapsed)

print("ONNX average latency:", sum(times)/len(times), "ms")