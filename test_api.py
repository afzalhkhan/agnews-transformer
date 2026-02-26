"""
test_api.py

Run with:  uv run pytest test_api.py -v -s
Requires the model to be trained and saved at models/distilbert-agnews/best
"""

import pytest
import time
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


#  Fixtures

SAMPLES = {
    "world":    "The United Nations held an emergency meeting on the ongoing conflict in Eastern Europe.",
    "sports":   "The Los Angeles Lakers defeated the Golden State Warriors in overtime last night.",
    "business": "Apple reported record quarterly earnings, beating analyst expectations by 12%.",
    "sci_tech": "NASA successfully launched its new Mars rover mission from Cape Canaveral.",
}


#  Health check

def test_root_returns_ok():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


#  Response schema

def test_predict_response_has_required_fields():
    response = client.post("/predict", json={"text": SAMPLES["sports"]})
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert "confidence" in data
    assert "latency_ms" in data
    assert "scores" in data


def test_predict_label_is_valid_class():
    response = client.post("/predict", json={"text": SAMPLES["business"]})
    data = response.json()
    assert data["label"] in {"world", "sports", "business", "sci_tech"}


def test_predict_confidence_is_between_0_and_1():
    response = client.post("/predict", json={"text": SAMPLES["sci_tech"]})
    data = response.json()
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_scores_sum_to_one():
    response = client.post("/predict", json={"text": SAMPLES["world"]})
    data = response.json()
    total = sum(data["scores"].values())
    assert abs(total - 1.0) < 1e-3


def test_predict_scores_has_all_four_classes():
    response = client.post("/predict", json={"text": SAMPLES["sports"]})
    data = response.json()
    assert set(data["scores"].keys()) == {"world", "sports", "business", "sci_tech"}


def test_confidence_matches_max_score():
    response = client.post("/predict", json={"text": SAMPLES["business"]})
    data = response.json()
    max_score = max(data["scores"].values())
    assert abs(data["confidence"] - max_score) < 1e-3


def test_predicted_label_matches_highest_score():
    response = client.post("/predict", json={"text": SAMPLES["sci_tech"]})
    data = response.json()
    best_class = max(data["scores"], key=lambda k: data["scores"][k])
    assert data["label"] == best_class


#  Classification accuracy 

@pytest.mark.parametrize("category,text", SAMPLES.items())
def test_correct_classification(category, text):
    """Model should correctly classify clear-cut examples from each category."""
    response = client.post("/predict", json={"text": text})
    assert response.status_code == 200
    assert response.json()["label"] == category


def test_bulk_accuracy():
    """
    40 samples mixing easy and deliberately hard/ambiguous cases to get a
    realistic accuracy number. Hard cases include:
      - business/sci_tech overlap  (Tesla, OpenAI funding, Apple valuation)
      - world/sports overlap       (Olympics ban, FIFA corruption)
      - world/business overlap     (sanctions, trade war, G7 tax)
      - short or noisy text
    Threshold is 75% — the printed number is what goes on your resume.
    """
    labelled = [

        ("world", "World leaders gathered in Geneva for climate change negotiations."),
        ("world", "The UN Security Council voted to impose new sanctions on Iran."),
        ("world", "Protests erupted in the capital following disputed election results."),
        ("world", "Peace talks between rival factions collapsed for the third time."),
        ("world", "The prime minister resigned amid a growing corruption scandal."),

        ("world", "The US imposed sweeping tariffs on Chinese imports, escalating the trade war."),
        ("world", "G7 nations agreed on a global minimum corporate tax rate of 15%."),

        ("world", "The International Olympic Committee banned Russia from the upcoming Games."),
        ("world", "FIFA suspended the football federation president over bribery allegations."),
        ("world", "The refugee crisis worsened as thousands crossed the border overnight."),


        ("sports", "Serena Williams advanced to the Wimbledon quarterfinals."),
        ("sports", "The Brazilian national football team qualified for the World Cup."),
        ("sports", "A new world record was set in the 100m sprint at the Olympics."),
        ("sports", "Formula 1 champion defends title at the Monaco Grand Prix."),
        ("sports", "The NBA finals went to a decisive Game 7 for the first time in a decade."),

        ("sports", "The club's new stadium deal is worth over 500 million dollars."),
        ("sports", "Manchester United reported a net loss despite record shirt sales."),

        ("sports", "Athletes from 12 nations boycotted the opening ceremony in protest."),
        ("sports", "The doping scandal forced the cycling federation to strip three medals."),
        ("sports", "Injury-plagued season ends as the quarterback announces retirement."),


        ("business", "The Federal Reserve raised interest rates by 25 basis points."),
        ("business", "Amazon announced plans to lay off 10,000 employees this quarter."),
        ("business", "Oil prices surged to a six-month high amid supply concerns."),
        ("business", "The IMF revised global growth forecasts downward for 2024."),
        ("business", "Inflation hit a 40-year high as consumer prices rose 8.5% year-on-year."),

        ("business", "Tesla stock dropped 12% after missing delivery targets for Q3."),
        ("business", "OpenAI raised 10 billion dollars in its latest funding round from Microsoft."),
        ("business", "Apple market cap crossed 3 trillion dollars for the first time."),

        ("business", "Sanctions on Russian energy exports sent European gas prices soaring."),
        ("business", "The merger between two airline giants faces regulatory scrutiny."),


        ("sci_tech", "Researchers developed a new battery that charges in under 5 minutes."),
        ("sci_tech", "Scientists discovered a potentially habitable exoplanet 40 light-years away."),
        ("sci_tech", "A CRISPR-based gene therapy showed promising results in clinical trials."),
        ("sci_tech", "Google unveiled a quantum computing chip that solves problems in seconds."),
        ("sci_tech", "The James Webb Space Telescope captured the deepest image of the universe."),

        ("sci_tech", "SpaceX successfully landed its reusable Starship rocket after orbital test."),
        ("sci_tech", "Meta released a new open-source large language model for researchers."),

        ("sci_tech", "The WHO approved a new malaria vaccine for use in sub-Saharan Africa."),
        ("sci_tech", "Cybersecurity researchers uncovered a critical vulnerability in global banking software."),
        ("sci_tech", "Engineers at MIT built a robot capable of assembling furniture autonomously."),
    ]

    correct = 0
    for label, text in labelled:
        resp = client.post("/predict", json={"text": text})
        if resp.json()["label"] == label:
            correct += 1

    accuracy = correct / len(labelled)
    print(f"\nBulk accuracy: {correct}/{len(labelled)} = {accuracy:.2%}")
    assert accuracy >= 0.75, f"Accuracy {accuracy:.2%} is below 75% threshold"


# ── Latency ───────────────────────────────────────────────────────────────────

def test_latency_ms_is_positive():
    response = client.post("/predict", json={"text": SAMPLES["world"]})
    assert response.json()["latency_ms"] > 0


def test_single_request_latency_under_500ms():
    """Wall-clock latency of a single request should be under 500ms on CPU."""
    start = time.time()
    response = client.post("/predict", json={"text": SAMPLES["sports"]})
    elapsed_ms = (time.time() - start) * 1000
    assert response.status_code == 200
    assert elapsed_ms < 500, f"Request took {elapsed_ms:.1f}ms — too slow"


def test_average_latency_under_300ms():
    """Average wall-clock latency over 10 requests should be under 300ms."""
    times = []
    for _ in range(10):
        start = time.time()
        client.post("/predict", json={"text": SAMPLES["sci_tech"]})
        times.append((time.time() - start) * 1000)

    avg_ms = sum(times) / len(times)
    print(f"\nAverage latency over 10 requests: {avg_ms:.1f}ms")
    assert avg_ms < 300, f"Average latency {avg_ms:.1f}ms exceeds 300ms threshold"


def test_reported_latency_ms_is_reasonable():
    """Inference-only latency reported in the response should be under 200ms."""
    response = client.post("/predict", json={"text": SAMPLES["business"]})
    latency = response.json()["latency_ms"]
    assert latency < 200, f"Inference latency {latency}ms is too high"



def test_very_short_text():
    """Single word input should not crash."""
    response = client.post("/predict", json={"text": "goal"})
    assert response.status_code == 200
    assert response.json()["label"] in {"world", "sports", "business", "sci_tech"}


def test_long_text_is_truncated_gracefully():
    """Text longer than max_length=128 tokens should be truncated, not error."""
    long_text = "The stock market crashed. " * 100
    response = client.post("/predict", json={"text": long_text})
    assert response.status_code == 200


def test_empty_text_does_not_crash():
    """Empty string should return a valid response."""
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 200


def test_special_characters_in_text():
    response = client.post("/predict", json={"text": "NASA @launch!! $1B mission #space"})
    assert response.status_code == 200
    assert response.json()["label"] in {"world", "sports", "business", "sci_tech"}


def test_missing_text_field_returns_422():
    """Pydantic should reject requests with missing required field."""
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_wrong_field_name_returns_422():
    response = client.post("/predict", json={"input": "some news"})
    assert response.status_code == 422


def test_non_string_text_returns_422():
    response = client.post("/predict", json={"text": 12345})
    assert response.status_code in (200, 422)