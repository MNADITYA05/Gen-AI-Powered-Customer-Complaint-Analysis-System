"""
Generates detailed evaluation reports from a trained ComplaintClassifier.
Kept separate so it can be called independently (e.g. in notebooks or CI).
"""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from core.analysis.classifier import ComplaintClassifier


class ModelEvaluator:
    def __init__(self, classifier: ComplaintClassifier):
        self._clf = classifier

    def full_report(self, df: pd.DataFrame) -> dict:
        """
        Run all three heads over `df` and return accuracy, per-class
        precision/recall/f1, and confusion matrices.
        """
        self._clf._assert_trained()

        texts = self._clf._preprocessor.clean_batch(df["complaint_text"].tolist())
        X = self._clf._vectorizer.transform(texts)

        cat_pred = self._clf._category_model.predict(X)
        emo_pred = self._clf._emotion_model.predict(X)
        sev_pred = self._clf._severity_model.predict(X)

        return {
            "category": self._head_report(df["category"], cat_pred,
                                           self._clf._category_model.classes_),
            "emotion":  self._head_report(df["emotion"],  emo_pred,
                                           self._clf._emotion_model.classes_),
            "severity": self._head_report(df["severity"], sev_pred,
                                           self._clf._severity_model.classes_),
        }

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _head_report(y_true, y_pred, classes) -> dict:
        classes = list(classes)
        cm = confusion_matrix(y_true, y_pred, labels=classes).tolist()
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        return {
            "accuracy":         round(accuracy_score(y_true, y_pred), 4),
            "confusion_matrix": {"labels": classes, "matrix": cm},
            "per_class":        {
                cls: {
                    "precision": round(report.get(cls, {}).get("precision", 0), 4),
                    "recall":    round(report.get(cls, {}).get("recall", 0), 4),
                    "f1_score":  round(report.get(cls, {}).get("f1-score", 0), 4),
                    "support":   int(report.get(cls, {}).get("support", 0)),
                }
                for cls in classes
            },
        }
