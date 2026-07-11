"""
Validates complaint text quality before storage or model training.
All thresholds are configurable at instantiation time.
"""
from __future__ import annotations

import re
from typing import Optional


_EXPLICIT_PROFANITY = frozenset(["fuck", "shit", "bitch", "asshole", "cunt"])

_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "ATM_FAILURE": [
        "atm", "machine", "card", "cash", "withdraw", "deposit",
        "screen", "error", "keypad", "transaction",
    ],
    "FRAUD_DETECTION": [
        "fraud", "unauthorized", "suspicious", "transaction", "charge",
        "account", "security", "stolen", "identity", "access",
    ],
    "UX_ISSUES": [
        "app", "website", "interface", "login", "crash", "slow",
        "error", "navigation", "portal", "banking",
    ],
}


class ValidationResult:
    def __init__(self, passed: bool, reason: str = ""):
        self.passed = passed
        self.reason = reason

    def __bool__(self) -> bool:
        return self.passed

    def __repr__(self) -> str:
        status = "PASS" if self.passed else f"FAIL: {self.reason}"
        return f"ValidationResult({status})"


class ComplaintValidator:
    def __init__(
        self,
        min_words: int = 10,
        max_words: int = 300,
        min_chars: int = 50,
        max_chars: int = 1500,
    ):
        self.min_words = min_words
        self.max_words = max_words
        self.min_chars = min_chars
        self.max_chars = max_chars

    def validate(
        self,
        text: str,
        category: Optional[str] = None,
    ) -> ValidationResult:
        """Run all checks in order; return on first failure."""
        checks = [
            self._check_not_empty,
            self._check_length,
            self._check_not_repetitive,
            self._check_profanity,
        ]
        if category:
            checks.append(lambda t: self._check_category_relevance(t, category))

        for check in checks:
            result = check(text)
            if not result:
                return result
        return ValidationResult(True)

    # ── Individual checks ─────────────────────────────────────────────────────

    def _check_not_empty(self, text: str) -> ValidationResult:
        if not text or not text.strip():
            return ValidationResult(False, "Text is empty")
        return ValidationResult(True)

    def _check_length(self, text: str) -> ValidationResult:
        words = len(text.split())
        chars = len(text)
        if words < self.min_words:
            return ValidationResult(False, f"Too short: {words} words (min {self.min_words})")
        if words > self.max_words:
            return ValidationResult(False, f"Too long: {words} words (max {self.max_words})")
        if chars < self.min_chars:
            return ValidationResult(False, f"Too short: {chars} chars (min {self.min_chars})")
        if chars > self.max_chars:
            return ValidationResult(False, f"Too long: {chars} chars (max {self.max_chars})")
        return ValidationResult(True)

    def _check_not_repetitive(self, text: str) -> ValidationResult:
        words = text.lower().split()
        if len(words) == 0:
            return ValidationResult(True)
        # Flag if any single word appears more than 20% of the time
        for word in set(words):
            if len(word) > 3 and words.count(word) / len(words) > 0.20:
                return ValidationResult(False, f"Repetitive word detected: '{word}'")
        return ValidationResult(True)

    def _check_profanity(self, text: str) -> ValidationResult:
        lower = text.lower()
        for word in _EXPLICIT_PROFANITY:
            # prefix match — catches "fucking", "shitty", etc.
            if re.search(rf"\b{re.escape(word)}", lower):
                return ValidationResult(False, "Contains explicit profanity")
        return ValidationResult(True)

    def _check_category_relevance(self, text: str, category: str) -> ValidationResult:
        keywords = _CATEGORY_KEYWORDS.get(category)
        if not keywords:
            return ValidationResult(True)  # unknown category — skip
        lower = text.lower()
        if not any(kw in lower for kw in keywords):
            return ValidationResult(
                False,
                f"No '{category}' keywords found in text"
            )
        return ValidationResult(True)
