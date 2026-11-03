"""Train Vietnamese YouTube sentiment classifiers using the THREE prepared CSV files.

Member 2 / data & baseline ML, 2026.

Your repository should contain:
    machine_learning/
      preprocessing.py      # exposes clean_text(text, segment=False)
      train_svm.py           # this script
      data/clean.csv         # full filtered corpus; NOT used for model fitting
      data/train.csv         # supervised training + cross-validation
      data/test.csv          # untouched holdout for reporting

PowerShell (from the repository root):
    .\\.venv\\Scripts\\python.exe machine_learning\\train_svm.py --model all \\
        --allow-unverified --out-dir machine_learning\\experiments\\final_pseudo

Outputs:
    <out-dir>/models/{svm,svm_word,lr,nb}_model.pkl  (as selected)
    <out-dir>/models/best_model.pkl
    <out-dir>/reports/model_comparison.csv
    <out-dir>/reports/*_metrics.json, *_errors.csv, *_confusion_matrix.png
    <out-dir>/reports/dataset_audit.json
    <out-dir>/predictions/video_sentiment_summary.csv
    <out-dir>/predictions/clean_predicted.csv

IMPORTANT: The provided datasets use pseudo/unverified labels. Their test scores
show agreement with those labels, NOT independently human-verified accuracy.
A balanced train/test split is useful for learning, but cannot estimate natural
population prevalence. For per-video percentages, predict on CLEAN (not on the
balanced splits), and only use rows with known video_id.

Model input is RAW comment text: the saved pipeline calls the team's
preprocessing.clean_text automatically. Keep that file alongside the model's
Python environment, unchanged after training. Load .pkl ONLY if trusted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

try:
    from preprocessing import clean_text
except ImportError as exc:
    raise SystemExit("Cannot import preprocessing.clean_text. Put preprocessing.py "
                     "next to train_svm.py and install its dependencies.") from exc

ROOT = Path(__file__).resolve().parent
LABELS = ["negative", "neutral", "positive"]
SEED = 42
ALIASES = {
    "negative": "negative", "neg": "negative", "0": "negative", "tiêu cực": "negative",
    "neutral": "neutral", "neu": "neutral", "1": "neutral", "trung lập": "neutral",
    "positive": "positive", "pos": "positive", "2": "positive", "tích cực": "positive",
}
HUMAN_STATUSES = {"human_verified", "manually_verified", "human_reviewed"}
VID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
MODEL_NAMES = {
    "svm": "Word + character TF-IDF / LinearSVC",
    "svm_word": "Word TF-IDF / LinearSVC",
    "lr": "Word TF-IDF / logistic (SGD)",
    "nb": "Word TF-IDF / MultinomialNB",
}


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_label(value: str) -> str | None:
    key = str(value).strip().lower()
    if key.endswith(".0") and key[:-2] in ("0", "1", "2"):
        key = key[:-2]
    return ALIASES.get(key)


def read_data(path: Path, require_label: bool = True) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError(f"Missing dataset: {path}")
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    if "text" not in df:
        raise ValueError(f"{path.name} needs a 'text' column; found: {df.columns.tolist()}")
    if require_label and "label" not in df:
        raise ValueError(f"{path.name} needs a 'label' column")
    if require_label:
        labels = df["label"].map(get_label)
        bad = df.loc[labels.isna(), "label"].value_counts().head(8).to_dict()
        if bad:
            raise ValueError(f"{path.name}: invalid/missing labels: {bad}")
        df["label"] = labels
    df["text"] = df["text"].astype(str).str.strip()
    df["video_id"] = df.get("video_id", pd.Series("", index=df.index)).astype(str).str.strip()
    # Use raw strings for YouTube IDs. Reading them as floats can corrupt identifiers.
    df["comment_id"] = df.get("comment_id", pd.Series("", index=df.index)).astype(str).str.strip()
    if not df["text"].ne("").all():
        raise ValueError(f"{path.name}: empty raw comments. Fix data before training.")
    df["_model_clean"] = [clean_text(s) for s in df["text"]]
    if df["_model_clean"].eq("").any():
        raise ValueError(f"{path.name}: empty text after preprocessing. Fix data.")
    return df


def count_labels(df: pd.DataFrame) -> dict:
    return {label: int((df["label"] == label).sum()) for label in LABELS}


def validate_splits(train: pd.DataFrame, test: pd.DataFrame, folds: int) -> dict:
    for name, df in (("TRAIN", train), ("TEST", test)):
        label_counts = count_labels(df)
        print(f"{name}: {len(df):,} comments | {label_counts}")
        if min(label_counts.values()) < (folds if name == "TRAIN" else 1):
            raise ValueError(f"{name} must contain all three labels with enough samples")
        dup_text = int(df["_model_clean"].duplicated().sum())
        if dup_text:
            raise ValueError(f"{name} contains {dup_text} duplicate cleaned comments. "
                             "Fix duplicates before cross-validation.")
        ids = df.loc[df["comment_id"].ne(""), "comment_id"]
        if ids.duplicated().any():
            raise ValueError(f"{name} contains duplicate nonempty comment_id values")

    overlap_texts = set(train["_model_clean"]) & set(test["_model_clean"])
    if overlap_texts:
        raise ValueError(f"DATA LEAKAGE: {len(overlap_texts)} identical cleaned texts in train/test")
    overlap_ids = (set(train["comment_id"]) - {""}) & (set(test["comment_id"]) - {""})
    if overlap_ids:
        raise ValueError(f"DATA LEAKAGE: {len(overlap_ids)} comment IDs in both train/test")
    known_train = set(train["video_id"]) - {""}
    known_test = set(test["video_id"]) - {""}
    overlap_vid = known_train & known_test
    if overlap_vid:
        raise ValueError(f"VIDEO LEAKAGE: {len(overlap_vid)} KNOWN video IDs "
                         "appear in train and test: " + ", ".join(sorted(overlap_vid)[:5]))
    print(f"LEAKAGE CHECK: no shared texts, comment IDs, or known video IDs. "
          f"Train known-video rows={int(train.video_id.ne('').sum())}; "
          f"test={int(test.video_id.ne('').sum())}")
    if train.video_id.eq("").any() or test.video_id.eq("").any():
        print("WARNING: video_id missing for many comments; can't guarantee "
              "full video-level disjointness for unknown IDs.")
    return {"train_rows": len(train), "test_rows": len(test),
            "train_labels": count_labels(train), "test_labels": count_labels(test),
            "known_video_ids_train": len(known_train),
            "known_video_ids_test": len(known_test),
            "unknown_video_rows_train": int(train.video_id.eq("").sum()),
            "unknown_video_rows_test": int(test.video_id.eq("").sum()),
            "overlap_clean_text": 0, "overlap_comment_id": 0,
            "overlap_known_video_id": 0}


def check_label_origin(train: pd.DataFrame, test: pd.DataFrame, allow: bool) -> dict:
    counts = {}
    for name, df in (("train", train), ("test", test)):
        if "review_status" not in df:
            counts[name] = {"missing_status": len(df)}
            if not allow:
                raise ValueError(f"{name}.csv has no review_status. "
                                 "Use --allow-unverified for pseudo-label experiments.")
            continue
        statuses = df["review_status"].str.strip().str.lower()
        counts[name] = statuses.value_counts().to_dict()
        if (~statuses.isin(HUMAN_STATUSES)).any() and not allow:
            raise ValueError(f"{name}.csv includes labels not explicitly human-verified. "
                             "Use --allow-unverified for experiments, or perform human review.")
    print("LABEL PROVENANCE:", counts)
    # Do not infer human verification from a stale 'verified' string or a confidence score.
    unverified = any(not set(values).issubset(HUMAN_STATUSES) for values in
                     (counts["train"], counts["test"]))
    if unverified:
        print("WARNING: TEST METRICS = agreement with pseudo/unverified labels, "
              "NOT validated real-world accuracy.")
    return {"status_counts": counts,
            "evaluation_type": ("pseudo_label_agreement_not_ground_truth_accuracy"
                                if unverified else "human_status_claim_not_independently_audited")}


def audit_clean(df: pd.DataFrame) -> dict:
    ids = df["video_id"]
    known = ids.ne("")
    invalid = ids[known & ~ids.str.fullmatch(VID_RE.pattern)]
    nonempty_comment_ids = df.loc[df.comment_id.ne(""), "comment_id"]
    report = {
        "rows": len(df), "label_counts": count_labels(df),
        "known_video_id_rows": int(known.sum()),
        "missing_video_id_rows": int((~known).sum()),
        "distinct_video_ids": int(ids[known].nunique()),
        "invalid_format_video_ids": int(invalid.shape[0]),
        "repeated_cleaned_texts": int(df["_model_clean"].duplicated().sum()),
        "duplicate_nonempty_comment_ids": int(nonempty_comment_ids.duplicated().sum()),
    }
    print("CLEAN AUDIT:", report)
    if report["invalid_format_video_ids"]:
        print("WARNING: some nonempty video IDs have invalid format; won't group them.")
    if "review_status" in df:
        report["review_status_counts"] = df.review_status.value_counts().to_dict()
    return report


def word_features():
    return TfidfVectorizer(
        preprocessor=partial(clean_text, segment=False),
        lowercase=False, token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2),
        min_df=2, max_df=0.97, max_features=120_000, sublinear_tf=True,
        strip_accents=None, dtype=np.float32,
    )


def char_features():
    return TfidfVectorizer(
        analyzer="char_wb", preprocessor=partial(clean_text, segment=False),
        ngram_range=(3, 5), min_df=2, max_features=160_000,
        sublinear_tf=True, lowercase=False, dtype=np.float32,
    )


def pipeline(kind: str) -> Pipeline:
    if kind == "svm":
        features = FeatureUnion([("word", word_features()), ("char", char_features())],
                                transformer_weights={"word": 1.0, "char": 0.8})
        clf = LinearSVC(C=1.0, max_iter=5000, random_state=SEED)
    elif kind == "lr":
        features = word_features()
        clf = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=1500,
                            tol=1e-3, random_state=SEED)
    elif kind == "svm_word":
        features = word_features()
        clf = LinearSVC(C=1.0, max_iter=5000, random_state=SEED)
    elif kind == "nb":
        features = word_features()
        clf = MultinomialNB(alpha=0.5)
    else:
        raise ValueError(f"Unknown model {kind}")
    return Pipeline([("features", features), ("clf", clf)])


def param_grid(kind: str) -> dict:
    if kind == "svm":
        return {"clf__C": [0.5, 1.0, 2.0, 4.0]}
    if kind == "lr":
        return {"clf__alpha": [1e-6, 1e-5, 1e-4],
                "features__ngram_range": [(1, 1), (1, 2)]}
    if kind == "svm_word":
        return {"clf__C": [0.5, 1.0, 2.0, 4.0],
                "features__ngram_range": [(1, 1), (1, 2)]}
    return {"clf__alpha": [0.1, 0.5, 1.0, 2.0],
            "features__ngram_range": [(1, 1), (1, 2)]}


def fit_one(kind: str, train: pd.DataFrame, cv, tune: bool, jobs: int):
    X, y = train["text"].tolist(), train["label"].tolist()
    base = pipeline(kind)
    start = time.perf_counter()
    if tune:
        search = GridSearchCV(base, param_grid(kind), cv=cv, scoring="f1_macro",
                              n_jobs=jobs, refit=True, error_score="raise")
        search.fit(X, y)
        model, score, params = search.best_estimator_, float(search.best_score_), search.best_params_
        std = float(search.cv_results_["std_test_score"][search.best_index_])
    else:
        scores = cross_val_score(base, X, y, cv=cv, scoring="f1_macro", n_jobs=jobs,
                                 error_score="raise")
        score, std = float(scores.mean()), float(scores.std())
        model, params = base.fit(X, y), {}
    return model, score, std, params, round(time.perf_counter() - start, 2)


def metrics_for(model: Pipeline, test: pd.DataFrame):
    pred = model.predict(test["text"].tolist())
    true = test["label"].tolist()
    prec, recall, f1, _ = precision_recall_fscore_support(
        true, pred, labels=LABELS, average="macro", zero_division=0)
    weighted_f1 = f1_score(true, pred, average="weighted", labels=LABELS, zero_division=0)
    mat = confusion_matrix(true, pred, labels=LABELS)
    report = classification_report(true, pred, labels=LABELS, output_dict=True,
                                   zero_division=0)
    return ({"accuracy": float(accuracy_score(true, pred)),
             "macro_precision": float(prec), "macro_recall": float(recall),
             "macro_f1": float(f1), "weighted_f1": float(weighted_f1),
             "confusion_matrix": mat.tolist(), "classification_report": report}, pred)


def save_matrix(cm: list, title: str, filename: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    counts = np.asarray(cm, dtype=int)
    totals = counts.sum(axis=1, keepdims=True)
    fraction = np.divide(counts, totals, out=np.zeros(counts.shape, dtype=float), where=totals > 0)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(fraction, vmin=0, vmax=1, cmap="Blues")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, f"{counts[i,j]:,}\n{fraction[i,j]:.1%}",
                    ha="center", va="center", color="white" if fraction[i,j] > .52 else "black")
    ax.set(xticks=range(3), yticks=range(3), xticklabels=LABELS,
           yticklabels=LABELS, xlabel="Predicted sentiment", ylabel="Source label", title=title)
    fig.colorbar(im, ax=ax, label="Fraction of true class", shrink=.8)
    fig.tight_layout()
    fig.savefig(filename, dpi=170)
    plt.close(fig)


def save_model(kind, model, info, metrics, pred, test, out: Path):
    model_dir, report_dir = out / "models", out / "reports"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{kind}_model.pkl"
    joblib.dump(model, model_path, compress=3)
    # Ensure serialized pipeline applies the same cleaning to RAW comments.
    reloaded = joblib.load(model_path)
    sample = test["text"].head(25).tolist()
    if not np.array_equal(reloaded.predict(sample), model.predict(sample)):
        raise RuntimeError(f"Model persistence test failed for {kind}")
    results = test.drop(columns=["_model_clean"], errors="ignore").copy()
    results["predicted_label"] = pred
    errors = results.loc[results["label"] != results["predicted_label"]]
    errors.to_csv(report_dir / f"{kind}_errors.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(metrics["confusion_matrix"], index=LABELS, columns=LABELS).to_csv(
        report_dir / f"{kind}_confusion_matrix.csv", encoding="utf-8-sig")
    save_matrix(metrics["confusion_matrix"], MODEL_NAMES[kind],
                report_dir / f"{kind}_confusion_matrix.png")
    (report_dir / f"{kind}_metrics.json").write_text(json.dumps({
        "model": MODEL_NAMES[kind], "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": info, "evaluation": metrics,
        "warning": "Unverified source labels; scores are pseudo-label agreement, not human-verified accuracy",
        "scikit_learn_version": sklearn.__version__,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Saved {model_path} ({model_path.stat().st_size/1e6:.1f} MB); "
          f"misclassified source labels: {len(errors):,}")
    return model_path


def analyze_clean(clean: pd.DataFrame, model, out: Path, batch_size: int = 4000):
    pred_dir = out / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    predictions = []
    raw = clean["text"].tolist()
    for i in range(0, len(raw), batch_size):
        predictions.extend(model.predict(raw[i:i + batch_size]).tolist())
    result = clean.drop(columns=["_model_clean"], errors="ignore").copy()
    result["predicted_label"] = predictions
    result.to_csv(pred_dir / "clean_predicted.csv", index=False, encoding="utf-8-sig")

    valid = result["video_id"].str.fullmatch(VID_RE.pattern)
    tagged = result.loc[valid, ["video_id", "predicted_label"]]
    table = pd.crosstab(tagged["video_id"], tagged["predicted_label"]).reindex(
        columns=LABELS, fill_value=0)
    table.index.name = "video_id"
    table["total_comments"] = table[LABELS].sum(axis=1)
    for lab in LABELS:
        table[f"{lab}_pct"] = (100 * table[lab] / table["total_comments"]).round(2)
    table = table.reset_index().sort_values("total_comments", ascending=False)
    table.to_csv(pred_dir / "video_sentiment_summary.csv", index=False, encoding="utf-8-sig")
    known = len(tagged)
    print(f"CLEAN PREDICTIONS: {len(result):,} comments predicted; "
          f"{known:,} have known valid video IDs across {len(table)} videos. "
          f"Missing/invalid IDs: {len(result) - known:,}.")
    print("WARNING: per-video percentages describe only OBSERVED known-ID comments; "
          "many comments have missing video IDs. No population-level guarantee.")


def main():
    parser = argparse.ArgumentParser(
        description="Vietnamese YouTube sentiment: best baseline from clean/train/test CSVs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--clean", type=Path, default=ROOT / "data/clean.csv",
                        help="Full filtered data, used only for inference/reporting (never training)")
    parser.add_argument("--train", type=Path, default=ROOT / "data/train.csv")
    parser.add_argument("--test", type=Path, default=ROOT / "data/test.csv")
    parser.add_argument("--model", choices=[*MODEL_NAMES, "all"], default="all")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "experiments/final_pseudo")
    parser.add_argument("--allow-unverified", action="store_true", help="Required for pseudo-label experiments")
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--tune", action="store_true", help="More expensive grid-search on TRAIN only")
    parser.add_argument("--jobs", type=int, default=1,
                        help="CV parallel processes; 1 uses lower RAM")
    parser.add_argument("--skip-clean-predictions", action="store_true",
                        help="Train/evaluate without predicting clean.csv")
    args = parser.parse_args()
    if args.cv_folds < 2:
        parser.error("--cv-folds must be >=2")
    out = args.out_dir
    (out / "reports").mkdir(parents=True, exist_ok=True)
    print("Reading prepared splits; NEVER merging clean.csv into training...")
    train, test = read_data(args.train), read_data(args.test)
    split_report = validate_splits(train, test, args.cv_folds)
    origin_report = check_label_origin(train, test, args.allow_unverified)
    clean = read_data(args.clean)
    clean_report = audit_clean(clean)
    # Ensure both splits are subsets of the reference corpus when IDs are present.
    clean_ids = set(clean.loc[clean.comment_id.ne(""), "comment_id"])
    not_in_clean = int((~train.loc[train.comment_id.ne(""), "comment_id"].isin(clean_ids)).sum()) \
        + int((~test.loc[test.comment_id.ne(""), "comment_id"].isin(clean_ids)).sum())
    if not_in_clean:
        print(f"WARNING: {not_in_clean} split comment IDs not found in clean.csv.")
    audit = {"inputs": {key: {"file": str(path), "sha256": sha256(path)}
                        for key, path in (("train", args.train), ("test", args.test), ("clean", args.clean))},
             "preprocessing_sha256": sha256(ROOT / "preprocessing.py") if (ROOT / "preprocessing.py").is_file() else None,
             "split": split_report, "labels": origin_report, "clean": clean_report,
             "split_ids_missing_from_clean": not_in_clean,
             "warning": "Video IDs absent for most data; pseudo labels cannot establish real-world accuracy."}
    (out / "reports/dataset_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=SEED)
    kinds = list(MODEL_NAMES) if args.model == "all" else [args.model]
    comparison = []
    trained = {}
    print("\nModel selection based only on TRAIN CV macro-F1. TEST remains held out.")
    for kind in kinds:
        print(f"\n=== {kind}: {MODEL_NAMES[kind]} ===", flush=True)
        model, score, std, params, elapsed = fit_one(kind, train, cv, args.tune, args.jobs)
        print(f"TRAIN CV macro-F1={score:.4f} +/- {std:.4f} | {elapsed}s | params={params}", flush=True)
        trained[kind] = (model, score, std, params, elapsed)
        comparison.append({"model_key": kind, "model": MODEL_NAMES[kind],
                           "train_cv_macro_f1": round(score, 5),
                           "train_cv_std": round(std, 5), "training_seconds": elapsed})
    winner = max(kinds, key=lambda k: trained[k][1])
    print(f"\nWINNER (TRAIN CV ONLY): {winner} / {MODEL_NAMES[winner]}")
    saved = {}
    for kind in kinds:
        model, score, std, params, elapsed = trained[kind]
        metrics, predictions = metrics_for(model, test)
        print(f"\nTEST {kind}: accuracy={metrics['accuracy']:.4f}, macro-F1={metrics['macro_f1']:.4f}, "
              f"weighted-F1={metrics['weighted_f1']:.4f}")
        print(classification_report(test.label, predictions, labels=LABELS, digits=4, zero_division=0))
        saved[kind] = save_model(kind, model,
                                 {"train_cv_macro_f1": score, "train_cv_std": std,
                                  "params": params, "seconds": elapsed},
                                 metrics, predictions, test, out)
        for entry in comparison:
            if entry["model_key"] == kind:
                entry.update({"test_accuracy": round(metrics["accuracy"], 5),
                              "test_macro_f1": round(metrics["macro_f1"], 5),
                              "test_weighted_f1": round(metrics["weighted_f1"], 5)})
    comparison_df = pd.DataFrame(comparison).sort_values("train_cv_macro_f1", ascending=False)
    comparison_df.to_csv(out / "reports/model_comparison.csv", index=False,
                         encoding="utf-8-sig")
    shutil.copyfile(saved[winner], out / "models/best_model.pkl")
    (out / "models/best_model_info.json").write_text(
        json.dumps({"selected_model": winner, "selected_by": "train_cv_macro_f1",
                    "train_cv_macro_f1": trained[winner][1],
                    "test_label_provenance": origin_report["evaluation_type"],
                    "classes": LABELS, "preprocessing_sha256": audit["preprocessing_sha256"]},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== Model comparison (ranked by TRAIN CV, NOT test performance) ===")
    print(comparison_df.to_string(index=False))
    if not args.skip_clean_predictions:
        analyze_clean(clean, trained[winner][0], out)
    print(f"\nReady: {out / 'models/best_model.pkl'}")
    print("Notes: metrics use UNVERIFIED pseudo labels; never present them as "
          "independently human-verified accuracy. Train/test should remain frozen.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as e:
        sys.exit(f"ERROR: {e}")
