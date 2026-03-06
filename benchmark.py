import time
import requests

url = "http://localhost:8000/predict"

payload = {
    "text": "NASA launches a new satellite to study climate change."
}

times = []

for _ in range(20):

    start = time.time()

    requests.post(url, json=payload)

    elapsed = (time.time() - start) * 1000
    times.append(elapsed)

avg = sum(times) / len(times)

print("Average latency:", round(avg, 2), "ms")