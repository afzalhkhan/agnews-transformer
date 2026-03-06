# Transformer-Based News Classification (AG News)

End-to-end machine learning project demonstrating **training, evaluation, deployment, and performance benchmarking** of transformer models for news topic classification.

The system classifies news articles into four categories:

* World
* Sports
* Business
* Sci/Tech

Models evaluated:

* DistilBERT
* BERT
* RoBERTa

---

## Architecture

Dataset → Transformer Training → Model Evaluation → FastAPI Inference API → Latency Benchmarking → ONNX Optimization

---

## Dataset

AG News dataset (120k training samples, 4 classes).

Each news article is classified into:

| Label | Category |
| ----- | -------- |
| 0     | World    |
| 1     | Sports   |
| 2     | Business |
| 3     | Sci/Tech |

---

## Model Training

Models fine-tuned using Hugging Face Transformers.

Training configuration:

* Batch size: 32
* Epochs: 1
* Max sequence length: 64
* Optimizer: AdamW
* Evaluation metric: Accuracy + Weighted F1

---

## Model Performance

| Model      | Accuracy | F1 Score |
| ---------- | -------- | -------- |
| DistilBERT | 0.9137   | 0.9134   |
| BERT       | 0.9225   | 0.9222   |
| RoBERTa    | 0.9185   | 0.9183   |

BERT achieves the highest accuracy, but inference latency varies significantly.

---

## Inference Latency Benchmark

Measured using a FastAPI inference service.

| Model      | Average Latency |
| ---------- | --------------- |
| DistilBERT | ~38 ms          |
| BERT       | ~113 ms         |
| RoBERTa    | ~100 ms         |

DistilBERT provides the best tradeoff between **accuracy and inference speed**.

---

## ONNX Optimization

The DistilBERT model was exported to ONNX and benchmarked using ONNX Runtime.

| Backend      | Latency |
| ------------ | ------- |
| PyTorch      | ~38 ms  |
| ONNX Runtime | ~60 ms  |

In this setup, PyTorch inference was faster for single-request CPU inference.

---

## API Deployment

The trained model is deployed using FastAPI.

Example request:

POST `/predict`

```
{
  "text": "NASA launches a new satellite to study climate change."
}
```

Example response:

```
{
  "label": "sci_tech",
  "confidence": 0.92,
  "latency_ms": 37.5,
  "scores": {
    "world": 0.02,
    "sports": 0.01,
    "business": 0.05,
    "sci_tech": 0.92
  }
}
```

Interactive API docs:

```
http://localhost:8000/docs
```

---

## Project Structure

```
transformer-news-classifier
│
├── api/
│   └── main.py
│
├── experiments/
│   └── run_experiments.py
│
├── models/
│   ├── distilbert/
│   ├── bert/
│   └── roberta/
│
├── benchmark.py
├── benchmark_onnx.py
├── export_onnx.py
└── README.md
```

---

## Key Features

* Transformer fine-tuning with Hugging Face
* Automated experiment pipeline
* Model comparison (DistilBERT vs BERT vs RoBERTa)
* FastAPI inference API
* Latency benchmarking
* ONNX model export and evaluation

---

## Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
* FastAPI
* ONNX Runtime
* Scikit-learn

---

## Future Improvements

* Batch inference support
* Quantization for faster CPU inference
* GPU deployment
* Model monitoring
