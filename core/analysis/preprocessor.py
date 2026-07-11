"""
Text preprocessing pipeline applied before vectorisation.
Keeping this isolated means it can be swapped without touching the classifier.
"""
from __future__ import annotations

import re
import string


class TextPreprocessor:
    """
    Lightweight text cleaner for banking complaint text.
    Does NOT stem or lemmatise — TF-IDF handles that implicitly via stop-words.
    """

    _CONTRACTION_MAP = {
        "can't": "cannot",
        "won't": "will not",
        "I'm": "I am",
        "I've": "I have",
        "I'll": "I will",
        "I'd": "I would",
        "don't": "do not",
        "doesn't": "does not",
        "didn't": "did not",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "hasn't": "has not",
        "haven't": "have not",
        "hadn't": "had not",
        "it's": "it is",
        "they're": "they are",
        "we're": "we are",
        "you're": "you are",
        "he's": "he is",
        "she's": "she is",
    }

    def clean(self, text: str) -> str:
        """Full preprocessing pipeline."""
        text = self._expand_contractions(text)
        text = self._remove_special_chars(text)
        text = self._normalise_whitespace(text)
        return text.lower().strip()

    def clean_batch(self, texts: list[str]) -> list[str]:
        return [self.clean(t) for t in texts]

    # ── Private steps ─────────────────────────────────────────────────────────

    def _expand_contractions(self, text: str) -> str:
        for contraction, expansion in self._CONTRACTION_MAP.items():
            text = re.sub(re.escape(contraction), expansion, text, flags=re.IGNORECASE)
        return text

    def _remove_special_chars(self, text: str) -> str:
        # Keep letters, digits, spaces, and basic punctuation useful for banking context
        allowed = set(string.ascii_letters + string.digits + " .,!?'-")
        return "".join(ch if ch in allowed else " " for ch in text)

    def _normalise_whitespace(self, text: str) -> str:
        return re.sub(r"\s+", " ", text)
