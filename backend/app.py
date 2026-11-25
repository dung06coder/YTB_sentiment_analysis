from pathlib import Path
import importlib.util
import sys
from typing import Any, Dict, List, Optional
from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ML_DIR = PROJECT_ROOT / "machine_learning"
for p in (str(PROJECT_ROOT), str(ML_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from backend.youtube_crawler import fetch_video_comments  # noqa: E402

try:
    from machine_learning.predict_phobert import PhoBertSentimentPredictor  # noqa: E402
except Exception:
    PhoBertSentimentPredictor = None  # type: ignore

app = Flask(__name__)
CORS(app)

SVM_MODEL_PATH = ML_DIR / "models" / "best_model.pkl"
PHOBERT_MODEL_DIR = ML_DIR / "models" / "phobert_finetuned"


class SentimentPredictorService:
    """Nạp và chạy suy luận cho mô hình SVM (.pkl) và PhoBERT."""

    def __init__(self) -> None:
        self.svm_model: Any = None
        self.phobert_predictor: Any = None
        self.loaded_models: Dict[str, bool] = {"svm": False, "phobert": False}
        self.load_errors: Dict[str, Optional[str]] = {"svm": None, "phobert": None}

    def load_models(self) -> None:
        try:
            if not SVM_MODEL_PATH.is_file():
                raise FileNotFoundError(f"Không tìm thấy file SVM: {SVM_MODEL_PATH}")
            self.svm_model = joblib.load(SVM_MODEL_PATH)
            self.loaded_models["svm"] = True
        except Exception as exc:
            self.loaded_models["svm"] = False
            self.load_errors["svm"] = str(exc)

        try:
            if PhoBertSentimentPredictor is None:
                raise ImportError("Không thể import PhoBertSentimentPredictor.")
            model_dir = PHOBERT_MODEL_DIR if PHOBERT_MODEL_DIR.exists() else None
            has_underthesea = importlib.util.find_spec("underthesea") is not None
            try:
                self.phobert_predictor = PhoBertSentimentPredictor(
                    model_dir=model_dir,
                    use_segment=has_underthesea,
                )
            except TypeError:
                self.phobert_predictor = PhoBertSentimentPredictor(model_dir=model_dir)
            self.loaded_models["phobert"] = True
        except Exception as exc:
            self.loaded_models["phobert"] = False
            self.load_errors["phobert"] = str(exc)

    def _predict_svm(self, comments: List[str]) -> List[str]:
        if not self.loaded_models["svm"] or self.svm_model is None:
            raise RuntimeError(f"Mô hình SVM chưa sẵn sàng: {self.load_errors['svm']}")
        preds = self.svm_model.predict(comments)
        return [str(label).strip().lower() for label in preds]

    def _predict_phobert(self, comments: List[str]) -> List[str]:
        if not self.loaded_models["phobert"] or self.phobert_predictor is None:
            raise RuntimeError(f"Mô hình PhoBERT chưa sẵn sàng: {self.load_errors['phobert']}")

        if hasattr(self.phobert_predictor, "predict_batch"):
            raw_preds = self.phobert_predictor.predict_batch(comments)
            results: List[str] = []
            for item in raw_preds:
                if isinstance(item, dict) and "label" in item:
                    results.append(str(item["label"]).strip().lower())
                else:
                    results.append(str(item).strip().lower())
            return results

        results = []
        for text in comments:
            res = self.phobert_predictor.predict_comment(text)
            label = res["label"] if isinstance(res, dict) and "label" in res else res
            results.append(str(label).strip().lower())
        return results

    def predict_batch(self, comments: List[str], model_name: str = "svm") -> List[str]:
        if not comments:
            return []
        normalized = model_name.strip().lower()
        if normalized == "svm":
            return self._predict_svm(comments)
        if normalized == "phobert":
            return self._predict_phobert(comments)
        raise ValueError(f"Mô hình '{model_name}' không được hỗ trợ. Chọn 'svm' hoặc 'phobert'.")


predictor = SentimentPredictorService()
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
    percentages = {k: round((v / total) * 100, 2) for k, v in counts.items()}
    return {
        "total": total,
        "counts": counts,
        "percentages": percentages,
    }


@app.get("/api/health")
def health_check():
    any_loaded = any(predictor.loaded_models.values())
    return jsonify({
        "status": "ok" if any_loaded else "degraded",
        "models": predictor.loaded_models,
        "errors": predictor.load_errors,
    }), 200


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
