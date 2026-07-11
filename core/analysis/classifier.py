"""
Three-head TF-IDF + Logistic Regression classifier.
Predicts category, emotion, and severity from raw complaint text.
MLflow is used to track every training run — runs are visible in `mlflow ui`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from core.analysis.preprocessor import TextPreprocessor
from core.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Labels that require specific ordering for display
SEVERITY_ORDER = ["low", "medium", "high", "critical"]


class ComplaintClassifier:
    """
    Trains three independent Logistic Regression heads on TF-IDF features:
        • category  (e.g. ATM_FAILURE, FRAUD_DETECTION, UX_ISSUES)
        • emotion   (e.g. frustrated, angry, worried)
        • severity  (low, medium, high, critical)

    All three share one vectoriser fitted on the training corpus.
    """

    MODEL_FILES = {
        "vectorizer":      "vectorizer.pkl",
        "category_model":  "category_model.pkl",
        "emotion_model":   "emotion_model.pkl",
        "severity_model":  "severity_model.pkl",
    }

    def __init__(self, model_dir: Optional[str] = None):
        self._model_dir = Path(model_dir or settings.model_dir)
        self._preprocessor = TextPreprocessor()

        self._vectorizer = TfidfVectorizer(
            max_features=8000,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self._category_model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        self._emotion_model  = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        self._severity_model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        self.is_trained = False

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame) -> dict:
        """
        Fit all three models on `df`.
        Requires columns: complaint_text, category, emotion, severity.
        Returns a metrics dict including the MLflow run_id.
        """
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)

        with mlflow.start_run() as run:
            texts = self._preprocessor.clean_batch(df["complaint_text"].tolist())
            X = self._vectorizer.fit_transform(texts)

            self._category_model.fit(X, df["category"])
            self._emotion_model.fit(X, df["emotion"])
            self._severity_model.fit(X, df["severity"])
            self.is_trained = True

            metrics = self._compute_metrics(X, df)

            # Log to MLflow
            mlflow.log_params({
                "vectorizer_max_features": self._vectorizer.max_features,
                "vectorizer_ngram_range":  str(self._vectorizer.ngram_range),
                "model_type":              "LogisticRegression",
                "training_samples":        len(df),
            })
            mlflow.log_metrics(metrics)

            self.save()
            mlflow.log_artifacts(str(self._model_dir), artifact_path="model_artifacts")

            run_id = run.info.run_id
            logger.info(
                "Training complete | cat=%.3f emo=%.3f sev=%.3f | run=%s",
                metrics["category_accuracy"],
                metrics["emotion_accuracy"],
                metrics["severity_accuracy"],
                run_id,
            )

        return {**metrics, "mlflow_run_id": run_id, "training_samples": len(df)}

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, text: str) -> dict:
        self._assert_trained()
        clean = self._preprocessor.clean(text)
        X = self._vectorizer.transform([clean])

        category = self._category_model.predict(X)[0]
        emotion  = self._emotion_model.predict(X)[0]
        severity = self._severity_model.predict(X)[0]

        cat_conf = float(self._category_model.predict_proba(X)[0].max())
        emo_conf = float(self._emotion_model.predict_proba(X)[0].max())
        sev_conf = float(self._severity_model.predict_proba(X)[0].max())

        return {
            "category": category,
            "emotion":  emotion,
            "severity": severity,
            "confidence": {
                "category": round(cat_conf, 4),
                "emotion":  round(emo_conf, 4),
                "severity": round(sev_conf, 4),
            },
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        return [self.predict(t) for t in texts]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        self._assert_trained()
        self._model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._vectorizer,      self._model_dir / self.MODEL_FILES["vectorizer"])
        joblib.dump(self._category_model,  self._model_dir / self.MODEL_FILES["category_model"])
        joblib.dump(self._emotion_model,   self._model_dir / self.MODEL_FILES["emotion_model"])
        joblib.dump(self._severity_model,  self._model_dir / self.MODEL_FILES["severity_model"])
        logger.info("Models saved to %s", self._model_dir)

    def load(self) -> bool:
        try:
            self._vectorizer     = joblib.load(self._model_dir / self.MODEL_FILES["vectorizer"])
            self._category_model = joblib.load(self._model_dir / self.MODEL_FILES["category_model"])
            self._emotion_model  = joblib.load(self._model_dir / self.MODEL_FILES["emotion_model"])
            self._severity_model = joblib.load(self._model_dir / self.MODEL_FILES["severity_model"])
            self.is_trained = True
            logger.info("Models loaded from %s", self._model_dir)
            return True
        except FileNotFoundError:
            return False

    # ── Introspection ─────────────────────────────────────────────────────────

    # Alias so API layer can call either .is_trained or .is_loaded uniformly
    @property
    def is_loaded(self) -> bool:
        return self.is_trained

    def get_info(self) -> dict:
        if not self.is_trained:
            return {"is_trained": False}
        return {
            "is_trained":       True,
            "category_classes": list(self._category_model.classes_),
            "emotion_classes":  list(self._emotion_model.classes_),
            "severity_classes": list(self._severity_model.classes_),
            "vectorizer_vocab_size": len(self._vectorizer.vocabulary_),
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _compute_metrics(self, X, df: pd.DataFrame) -> dict:
        cat_pred = self._category_model.predict(X)
        emo_pred = self._emotion_model.predict(X)
        sev_pred = self._severity_model.predict(X)
        return {
            "category_accuracy": round(accuracy_score(df["category"], cat_pred), 4),
            "emotion_accuracy":  round(accuracy_score(df["emotion"],  emo_pred),  4),
            "severity_accuracy": round(accuracy_score(df["severity"], sev_pred),  4),
        }

    def _assert_trained(self) -> None:
        if not self.is_trained:
            raise RuntimeError(
                "Classifier not trained. Call train() or load() first."
            )
