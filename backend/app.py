from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ML_DIR = PROJECT_ROOT / "machine_learning"
for p in (str(PROJECT_ROOT), str(ML_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(Path(__file__).resolve().parent / ".env")

from backend.youtube_crawler import YouTubeCrawlerError, fetch_video_comments  # noqa: E402

try:
    from machine_learning.preprocessing import clean_text  # noqa: E402
except Exception:
    clean_text = None  # type: ignore

try:
    from machine_learning.predict_phobert import PhoBertSentimentPredictor  # noqa: E402
except Exception:
    PhoBertSentimentPredictor = None  # type: ignore


app = Flask(__name__)
CORS(app)

SVM_MODEL_PATH = ML_DIR / "models" / "best_model.pkl"
PHOBERT_MODEL_DIR = ML_DIR / "models" / "phobert_finetuned"
REPORTS_DIR = ML_DIR / "reports"

# Cờ cấu hình linh hoạt qua biến môi trường
SVM_USE_PREPROCESS = os.getenv("SVM_USE_PREPROCESS", "true").lower() == "true"
SVM_USE_SEGMENT = os.getenv("SVM_USE_SEGMENT", "false").lower() == "true"
ALLOW_PHOBERT_FALLBACK = os.getenv("ALLOW_PHOBERT_BASE_FALLBACK", "false").lower() == "true"

SUPPORTED_MODELS = ("svm", "phobert")
VALID_LABELS = ("positive", "neutral", "negative")

# Chỉ số đánh giá mặc định (fallback khi không đọc được file JSON trong reports/)
DEFAULT_METRICS: Dict[str, Dict[str, Any]] = {
    "svm": {
        "accuracy": 0.7442,
        "precision": 0.7444,
        "recall": 0.7442,
        "f1": 0.7440,
        "f1_per_class": {"positive": 0.7840, "neutral": 0.6988, "negative": 0.7491},
    },
    "phobert": {
        "accuracy": 0.7916,
        "precision": 0.7916,
        "recall": 0.7916,
        "f1": 0.7901,
        "f1_per_class": {"positive": 0.8243, "neutral": 0.7427, "negative": 0.8034},
    },
}


def normalize_label(raw_label: Any) -> str:
    """Chuẩn hóa nhãn đầu ra về 3 giá trị: positive, neutral, negative."""
    text = str(raw_label).strip().lower()
    mapping = {
        "positive": "positive",
        "pos": "positive",
        "tích cực": "positive",
        "tich cuc": "positive",
        "2": "positive",
        "neutral": "neutral",
        "neu": "neutral",
        "trung lập": "neutral",
        "trung lap": "neutral",
        "1": "neutral",
        "negative": "negative",
        "neg": "negative",
        "tiêu cực": "negative",
        "tieu cuc": "negative",
        "0": "negative",
    }
    return mapping.get(text, "neutral")


def softmax(scores: np.ndarray) -> np.ndarray:
    """Chuyển điểm decision_function thành xác suất giả (0..1)."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim == 1:
        # Trường hợp nhị phân: chuyển bằng sigmoid
        probs_pos = 1.0 / (1.0 + np.exp(-np.clip(scores, -50, 50)))
        return np.vstack([1.0 - probs_pos, probs_pos]).T
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_vals = np.exp(np.clip(shifted, -50, 50))
    return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)


def load_model_metrics(report_filename: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    """Đọc chỉ số đánh giá từ machine_learning/reports/*.json, fallback nếu khác schema."""
    report_path = REPORTS_DIR / report_filename
    if not report_path.is_file():
        return fallback

    try:
        with report_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return fallback

    # Lấy các chỉ số tổng quát từ nhiều kiểu tên trường phổ biến
    weighted = raw.get("weighted avg") or raw.get("classification_report", {}).get("weighted avg") or {}
    macro = raw.get("macro avg") or raw.get("classification_report", {}).get("macro avg") or {}

    def pick_float(keys: List[str], default_val: float) -> float:
        for k in keys:
            if k in raw and isinstance(raw[k], (int, float)):
                return round(float(raw[k]), 4)
        return default_val

    accuracy = pick_float(["accuracy", "test_accuracy"], fallback["accuracy"])
    precision = pick_float(
        ["precision", "precision_weighted", "weighted_precision", "macro_precision"],
        round(float(weighted.get("precision", macro.get("precision", fallback["precision"]))), 4),
    )
    recall = pick_float(
        ["recall", "recall_weighted", "weighted_recall", "macro_recall"],
        round(float(weighted.get("recall", macro.get("recall", fallback["recall"]))), 4),
    )
    f1 = pick_float(
        ["f1", "f1_score", "f1_weighted", "weighted_f1", "macro_f1", "f1_macro"],
        round(float(weighted.get("f1-score", macro.get("f1-score", fallback["f1"]))), 4),
    )

    # Lấy F1 theo từng lớp
    f1_per_class = dict(fallback["f1_per_class"])
    raw_f1_cls = raw.get("f1_per_class") or raw.get("per_class_f1") or {}
    cls_report = raw.get("classification_report") or raw

    for lbl in VALID_LABELS:
        if lbl in raw_f1_cls and isinstance(raw_f1_cls[lbl], (int, float)):
            f1_per_class[lbl] = round(float(raw_f1_cls[lbl]), 4)
        elif isinstance(cls_report.get(lbl), dict) and "f1-score" in cls_report[lbl]:
            f1_per_class[lbl] = round(float(cls_report[lbl]["f1-score"]), 4)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f1_per_class": f1_per_class,
    }


class SentimentPredictorService:
    """Nạp và chạy suy luận cho mô hình SVM (.pkl) và PhoBERT."""

    def __init__(self) -> None:
        self.svm_model: Any = None
        self.phobert_predictor: Any = None
        self.loaded_models: Dict[str, bool] = {"svm": False, "phobert": False}
        self.load_errors: Dict[str, Optional[str]] = {"svm": None, "phobert": None}
        self.metrics: Dict[str, Dict[str, Any]] = {
            "svm": load_model_metrics("svm_metrics.json", DEFAULT_METRICS["svm"]),
            "phobert": load_model_metrics("phobert_metrics.json", DEFAULT_METRICS["phobert"]),
        }

    def load_models(self) -> None:
        # 1. Nạp SVM
        try:
            if not SVM_MODEL_PATH.is_file():
                raise FileNotFoundError(f"Không tìm thấy file SVM: {SVM_MODEL_PATH}")
            self.svm_model = joblib.load(SVM_MODEL_PATH)
            self.loaded_models["svm"] = True
            self.load_errors["svm"] = None
        except Exception as exc:
            self.loaded_models["svm"] = False
            self.load_errors["svm"] = str(exc)

        # 2. Nạp PhoBERT (chặn fallback ngầm sang model chưa fine-tune)
        try:
            if PhoBertSentimentPredictor is None:
                raise ImportError("Không thể import PhoBertSentimentPredictor (kiểm tra torch/transformers).")

            has_finetuned = (PHOBERT_MODEL_DIR / "config.json").is_file()
            if not has_finetuned and not ALLOW_PHOBERT_FALLBACK:
                raise FileNotFoundError(
                    f"Chưa có trọng số PhoBERT fine-tuned tại '{PHOBERT_MODEL_DIR}'. "
                    "Hãy tải thư mục phobert_finetuned về máy (hoặc bật ALLOW_PHOBERT_BASE_FALLBACK=true để test)."
                )

            model_dir = PHOBERT_MODEL_DIR if has_finetuned else None
            has_segmenter = (
                importlib.util.find_spec("underthesea") is not None
                or importlib.util.find_spec("pyvi") is not None
            )
            self.phobert_predictor = PhoBertSentimentPredictor(
                model_dir=model_dir,
                use_segment=has_segmenter,
            )
            self.loaded_models["phobert"] = True
            self.load_errors["phobert"] = None
        except Exception as exc:
            self.loaded_models["phobert"] = False
            self.load_errors["phobert"] = str(exc)

    def _preprocess_for_svm(self, texts: List[str]) -> List[str]:
        if not SVM_USE_PREPROCESS or clean_text is None:
            return texts

        cleaned_list: List[str] = []
        for t in texts:
            try:
                c = clean_text(t, segment=SVM_USE_SEGMENT)
            except TypeError:
                c = clean_text(t)
            cleaned_list.append(c if c and c.strip() else t)
        return cleaned_list

    def _predict_svm(self, texts: List[str]) -> List[Dict[str, Any]]:
        if not self.loaded_models["svm"] or self.svm_model is None:
            raise RuntimeError(f"Mô hình SVM chưa sẵn sàng: {self.load_errors['svm']}")

        processed = self._preprocess_for_svm(texts)
        raw_preds = self.svm_model.predict(processed)

        confidences: List[float]
        if hasattr(self.svm_model, "predict_proba"):
            probs = self.svm_model.predict_proba(processed)
            confidences = [round(float(np.max(p)), 4) for p in probs]
        elif hasattr(self.svm_model, "decision_function"):
            margins = self.svm_model.decision_function(processed)
            probs = softmax(margins)
            confidences = [round(float(np.max(p)), 4) for p in probs]
        else:
            confidences = [0.85 for _ in raw_preds]

        return [
            {"label": normalize_label(lbl), "confidence": conf}
            for lbl, conf in zip(raw_preds, confidences)
        ]

    def _predict_phobert(self, texts: List[str]) -> List[Dict[str, Any]]:
        if not self.loaded_models["phobert"] or self.phobert_predictor is None:
            raise RuntimeError(f"Mô hình PhoBERT chưa sẵn sàng: {self.load_errors['phobert']}")

        # PhoBertSentimentPredictor đã tự gọi clean_text bên trong
        if hasattr(self.phobert_predictor, "predict_batch"):
            raw_preds = self.phobert_predictor.predict_batch(texts)
        else:
            raw_preds = [self.phobert_predictor.predict_comment(t) for t in texts]

        results: List[Dict[str, Any]] = []
        for item in raw_preds:
            if isinstance(item, dict):
                lbl = normalize_label(item.get("label", "neutral"))
                conf = round(float(item.get("confidence", 0.5)), 4)
            else:
                lbl = normalize_label(item)
                conf = 0.5
            results.append({"label": lbl, "confidence": conf})
        return results

    def predict(self, texts: List[str], model_name: str) -> List[Dict[str, Any]]:
        if not texts:
            return []
        name = model_name.strip().lower()
        if name == "svm":
            return self._predict_svm(texts)
        if name == "phobert":
            return self._predict_phobert(texts)
        raise ValueError(f"Mô hình '{model_name}' không được hỗ trợ.")


def build_model_result(
    comment_items: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    sample_limit: int = 12,
) -> Dict[str, Any]:
    """Đóng gói kết quả cho 1 mô hình đúng chuẩn mockResult.json."""
    total = len(predictions)
    counts = {lbl: 0 for lbl in VALID_LABELS}
    total_conf = 0.0

    buckets: Dict[str, List[Dict[str, Any]]] = {lbl: [] for lbl in VALID_LABELS}

    for item, pred in zip(comment_items, predictions):
        lbl = pred["label"]
        conf = float(pred["confidence"])
        counts[lbl] = counts.get(lbl, 0) + 1
        total_conf += conf

        buckets[lbl].append(
            {
                "text": item["text"],
                "label": lbl,
                "confidence": round(conf, 4),
                "author": item.get("author", "Ẩn danh"),
                "published_at": item.get("published_at", ""),
                "like_count": int(item.get("like_count", 0)),
            }
        )

    # Chọn mẫu cân bằng giữa 3 cảm xúc (ưu tiên bình luận có confidence cao)
    samples: List[Dict[str, Any]] = []
    for lbl in VALID_LABELS:
        buckets[lbl].sort(key=lambda x: x["confidence"], reverse=True)

    idx = 0
    while len(samples) < sample_limit:
        added = False
        for lbl in VALID_LABELS:
            if idx < len(buckets[lbl]) and len(samples) < sample_limit:
                samples.append(buckets[lbl][idx])
                added = True
        if not added:
            break
        idx += 1

    percentages = {
        lbl: round((counts[lbl] / total) * 100, 1) if total > 0 else 0.0
        for lbl in VALID_LABELS
    }
    avg_confidence = round((total_conf / total) * 100, 1) if total > 0 else 0.0

    return {
        "positive": percentages["positive"],
        "neutral": percentages["neutral"],
        "negative": percentages["negative"],
        "counts": counts,
        "avg_confidence": avg_confidence,
        "metrics": metrics,
        "samples": samples,
    }


predictor = SentimentPredictorService()
predictor.load_models()


def error_response(code: str, message: str, status_code: int = 400):
    return jsonify({"error": code, "message": message}), status_code


@app.get("/api/health")
def health_check():
    any_loaded = any(predictor.loaded_models.values())
    return jsonify(
        {
            "status": "ok" if any_loaded else "degraded",
            "models": predictor.loaded_models,
            "errors": predictor.load_errors,
        }
    ), 200


@app.post("/api/analyze")
def analyze_comments():
    start_time = time.perf_counter()
    payload = request.get_json(silent=True) or {}

    # 1. Xác thực danh sách mô hình yêu cầu (hỗ trợ cả `models: [...]` lẫn `model: "..."`)
    raw_models = payload.get("models")
    if raw_models is None:
        single_model = payload.get("model", "svm")
        raw_models = [single_model]
    if not isinstance(raw_models, list) or not raw_models:
        return error_response("INVALID_MODEL", "Vui lòng chọn ít nhất một mô hình hợp lệ.", 400)

    requested_models: List[str] = []
    for m in raw_models:
        m_norm = str(m).strip().lower()
        if m_norm not in SUPPORTED_MODELS:
            return error_response(
                "INVALID_MODEL",
                f"Mô hình '{m}' không được hỗ trợ. Chỉ hỗ trợ: {', '.join(SUPPORTED_MODELS)}.",
                400,
            )
        if m_norm not in requested_models:
            requested_models.append(m_norm)

    # Kiểm tra model đã nạp sẵn sàng TRƯỚC khi tốn quota cào YouTube
    not_ready = [m for m in requested_models if not predictor.loaded_models.get(m)]
    if not_ready:
        details = "; ".join(f"{m}: {predictor.load_errors.get(m)}" for m in not_ready)
        return error_response(
            "MODEL_NOT_READY",
            f"Mô hình chưa sẵn sàng trên server ({details}).",
            503,
        )

    # 2. Xác thực max_comments
    try:
        max_comments = int(payload.get("max_comments", 500))
        if max_comments <= 0:
            raise ValueError
    except (TypeError, ValueError):
        return error_response("INVALID_PARAM", "'max_comments' phải là số nguyên dương.", 400)

    # 3. Lấy dữ liệu bình luận (từ YouTube URL hoặc mảng comments truyền tay để test)
    try:
        if isinstance(payload.get("comments"), list):
            video_id = "manual_input"
            comment_items: List[Dict[str, Any]] = []
            for item in payload["comments"]:
                if isinstance(item, dict) and str(item.get("text", "")).strip():
                    comment_items.append(
                        {
                            "text": str(item["text"]).strip(),
                            "author": str(item.get("author", "Người dùng")),
                            "published_at": str(item.get("published_at", "")),
                            "like_count": int(item.get("like_count", 0) or 0),
                        }
                    )
                elif isinstance(item, str) and item.strip():
                    comment_items.append(
                        {
                            "text": item.strip(),
                            "author": "Người dùng",
                            "published_at": "",
                            "like_count": 0,
                        }
                    )
            if not comment_items:
                return error_response("NO_COMMENTS", "Danh sách bình luận trống.", 400)
        else:
            url = str(payload.get("url", "")).strip()
            if not url:
                return error_response("INVALID_URL", "Thiếu link YouTube cần phân tích.", 400)
            crawled = fetch_video_comments(url=url, max_comments=max_comments)
            video_id = crawled["video_id"]
            comment_items = crawled["comments"]

        # 4. Chạy suy luận song song/tuần tự trên cùng 1 tập bình luận
        texts = [c["text"] for c in comment_items]
        results: Dict[str, Any] = {}
        for model_name in requested_models:
            preds = predictor.predict(texts, model_name)
            results[model_name] = build_model_result(
                comment_items=comment_items,
                predictions=preds,
                metrics=predictor.metrics[model_name],
            )

        latency_ms = int((time.perf_counter() - start_time) * 1000)

        return jsonify(
            {
                "video_id": video_id,
                "total_comments": len(comment_items),
                "latency_ms": latency_ms,
                "results": results,
            }
        ), 200

    except YouTubeCrawlerError as exc:
        return error_response(exc.code, exc.message, exc.status_code)
    except Exception as exc:
        return error_response("UNKNOWN", f"Lỗi xử lý hệ thống: {exc}", 500)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    # Tắt use_reloader để tránh nạp mô hình ML 2 lần vào RAM
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)