"""
core/analysis/multi_task_model.py
----------------------------------
Multi-task RoBERTa-base classifier with three shared-encoder heads:
  - category  (ATM_FAILURE / FRAUD_DETECTION / UX_ISSUES)
  - severity  (low / medium / high / critical)
  - emotion   (angry / anxious / confused / frustrated)

Backbone: roberta-base (benchmark winner — avg_acc 75.4% vs DistilBERT 66.3%)
Trained on Colab Pro T4 GPU via train_multitask_colab.ipynb.
Falls back to classical TF-IDF + RF if weights are not present.

Improvements over v1 (targeting 90% avg accuracy):
  - Mean pooling over all tokens (vs CLS-only)
  - AutoTokenizer (correct RoBERTa BPE tokenizer, not DistilBERT WordPiece)
  - MAX_LEN=256 (captures full complaint text)
  - Trained on full dataset (not 2000-sample subset)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Files expected inside models/multitask_model/
_MODEL_WEIGHTS = "pytorch_model.pt"
_LABEL_MAPS    = "label_maps.json"
_CONFIG        = "config.json"


# ── PyTorch model definition (must match Colab notebook exactly) ──────────────

class _MultiTaskNet(nn.Module):
    """
    RoBERTa-base encoder + 3 linear classification heads.
    Uses mean pooling over all non-padding tokens for richer representations.
    """

    def __init__(self, n_cat: int, n_sev: int, n_emo: int, base_model: str = "roberta-base"):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(base_model)
        self.dropout  = nn.Dropout(0.3)
        hidden = self.backbone.config.hidden_size  # 768
        self.cat_head = nn.Linear(hidden, n_cat)
        self.sev_head = nn.Linear(hidden, n_sev)
        self.emo_head = nn.Linear(hidden, n_emo)

    @staticmethod
    def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean-pool token embeddings, ignoring padding tokens."""
        mask      = attention_mask.unsqueeze(-1).float()          # (B, L, 1)
        summed    = (last_hidden_state * mask).sum(dim=1)         # (B, H)
        counts    = mask.sum(dim=1).clamp(min=1e-9)               # (B, 1)
        return summed / counts                                     # (B, H)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out    = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._mean_pool(out.last_hidden_state, attention_mask)
        pooled = self.dropout(pooled)
        return self.cat_head(pooled), self.sev_head(pooled), self.emo_head(pooled)


# ── Public wrapper ─────────────────────────────────────────────────────────────

class MultiTaskClassifier:
    """
    Loads the fine-tuned multi-task RoBERTa-base model and exposes
    the same predict() / predict_batch() / get_info() interface as
    ComplaintClassifier so the API layer needs zero changes.
    """

    def __init__(self, model_dir: str | Path):
        self._dir    = Path(model_dir) / "multitask_model"
        self._model: Optional[_MultiTaskNet] = None
        self._tokenizer = None
        self._label_maps: dict = {}
        self._config: dict     = {}
        self.is_loaded = False

    @property
    def is_trained(self) -> bool:
        """Alias for is_loaded — duck-typing compatibility with ComplaintClassifier."""
        return self.is_loaded

    # ── Load ──────────────────────────────────────────────────────────────────

    def load(self) -> bool:
        """Return True if model files exist and loaded successfully."""
        weights_path = self._dir / _MODEL_WEIGHTS
        labels_path  = self._dir / _LABEL_MAPS
        config_path  = self._dir / _CONFIG

        if not (weights_path.exists() and labels_path.exists() and config_path.exists()):
            return False

        try:
            from transformers import AutoTokenizer

            with open(config_path) as f: self._config     = json.load(f)
            with open(labels_path) as f: self._label_maps = json.load(f)

            base_model = self._config.get("base_model", "roberta-base")

            # AutoTokenizer picks the correct tokenizer for any backbone
            self._tokenizer = AutoTokenizer.from_pretrained(base_model)

            self._model = _MultiTaskNet(
                n_cat=self._config["n_category"],
                n_sev=self._config["n_severity"],
                n_emo=self._config["n_emotion"],
                base_model=base_model,
            )
            state = torch.load(weights_path, map_location="cpu")
            self._model.load_state_dict(state)
            self._model.eval()

            self.is_loaded = True
            logger.info(
                "MultiTask RoBERTa-base loaded from %s  (avg_acc=%.4f)",
                self._dir,
                self._config.get("best_avg_accuracy", 0),
            )
            return True

        except Exception as exc:
            logger.error("Failed to load multi-task model: %s", exc)
            return False

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, text: str) -> dict:
        if not self.is_loaded:
            raise RuntimeError("MultiTaskClassifier not loaded. Call load() first.")

        enc = self._tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=self._config.get("max_length", 256),
            return_tensors="pt",
        )

        with torch.no_grad():
            logits_c, logits_s, logits_e = self._model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
            )

        probs_c = torch.softmax(logits_c, dim=-1)[0]
        probs_s = torch.softmax(logits_s, dim=-1)[0]
        probs_e = torch.softmax(logits_e, dim=-1)[0]

        cat_idx = probs_c.argmax().item()
        sev_idx = probs_s.argmax().item()
        emo_idx = probs_e.argmax().item()

        # Confidence floor: if severity confidence < 50%, default to "medium"
        # rather than committing to a wrong low-confidence prediction.
        if probs_s[sev_idx].item() < 0.50:
            fallback = "medium"
            if fallback in self._label_maps["severity"]:
                sev_idx = self._label_maps["severity"].index(fallback)

        return {
            "category": self._label_maps["category"][cat_idx],
            "severity": self._label_maps["severity"][sev_idx],
            "emotion":  self._label_maps["emotion"][emo_idx],
            "confidence": {
                "category": round(probs_c[cat_idx].item(), 4),
                "severity": round(probs_s[sev_idx].item(), 4),
                "emotion":  round(probs_e[emo_idx].item(), 4),
            },
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        return [self.predict(t) for t in texts]

    # ── Info ──────────────────────────────────────────────────────────────────

    def get_info(self) -> dict:
        if not self.is_loaded:
            return {"is_loaded": False, "is_trained": False}
        return {
            "is_loaded":         True,
            "is_trained":        True,          # duck-typing alias for ModelInfoResponse
            "model_type":        "MultiTask RoBERTa-base",
            "base_model":        self._config.get("base_model"),
            "best_avg_accuracy": self._config.get("best_avg_accuracy"),
            "category_classes":  self._label_maps.get("category", []),
            "severity_classes":  self._label_maps.get("severity", []),
            "emotion_classes":   self._label_maps.get("emotion",  []),
        }
