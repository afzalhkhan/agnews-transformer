import onnxruntime as ort
import numpy as np


session = ort.InferenceSession("distilbert.onnx")


def predict(input_ids, attention_mask):

    outputs = session.run(
        None,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
    )

    logits = outputs[0]

    exp = np.exp(logits)
    probs = exp / np.sum(exp, axis=1, keepdims=True)

    return probs