from typing import Any, Dict, List
from flask import Flask, jsonify, request
from flask_cors import CORS

from backend.youtube_crawler import fetch_video_comments

app = Flask(__name__)
CORS(app)


class SentimentPredictorSkeleton:
    """Giao diện chờ nạp mô hình SVM (best_model.pkl) và PhoBERT."""

    def __init__(self) -> None:
        self.loaded = False

    def load_models(self) -> None:
        # TODO: Nạp best_model.pkl (SVM) và checkpoint PhoBERT 1 lần khi khởi động
        self.loaded = True

    def predict_batch(self, comments: List[str], model_name: str = "svm") -> List[str]:
        # Tạm thời trả nhãn mẫu ("positive", "neutral", "negative") để test luồng API
        labels = ["positive", "neutral", "negative"]
        return [labels[i % len(labels)] for i in range(len(comments))]


predictor = SentimentPredictorSkeleton()
predictor.load_models()


def summarize_sentiments(labels: List[str]) -> Dict[str, Any]:
    total = len(labels)
    if total == 0:
        return {
            "total": 0,
            "counts": {"positive": 0, "neutral": 0, "negative": 0},
            "percentages": {"positive": 0.0, "neutral": 0.0, "negative": 0.0},
        }

    counts = {
        "positive": labels.count("positive"),
        "neutral": labels.count("neutral"),
        "negative": labels.count("negative"),
    }
    percentages = {
        k: round((v / total) * 100, 2) for k, v in counts.items()
    }
    return {
        "total": total,
        "counts": counts,
        "percentages": percentages,
    }


@app.get("/api/health")
def health_check():
    return jsonify({"status": "ok", "models_loaded": predictor.loaded}), 200


@app.post("/api/analyze")
def analyze_comments():
    payload = request.get_json(silent=True) or {}
    model_name = str(payload.get("model", "svm")).lower()
    max_comments = int(payload.get("max_comments", 500))

    try:
        if isinstance(payload.get("comments"), list):
            comments = [str(c).strip() for c in payload["comments"] if str(c).strip()]
        else:
            url = str(payload.get("url", "")).strip()
            if not url:
                return jsonify({"error": "Thiếu 'url' video YouTube hoặc danh sách 'comments'."}), 400
            comments = fetch_video_comments(url=url, max_comments=max_comments)

        predictions = predictor.predict_batch(comments=comments, model_name=model_name)
        summary = summarize_sentiments(predictions)

        samples = [
            {"comment": c, "sentiment": s}
            for c, s in zip(comments[:10], predictions[:10])
        ]

        return jsonify({
            "model": model_name,
            "summary": summary,
            "samples": samples,
        }), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Lỗi xử lý hệ thống: {exc}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
