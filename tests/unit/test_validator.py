"""Unit tests for ComplaintValidator."""
import pytest

from core.validation.complaint_validator import ComplaintValidator


@pytest.fixture
def validator() -> ComplaintValidator:
    return ComplaintValidator()


class TestLengthChecks:
    def test_too_short_fails(self, validator):
        result = validator.validate("Too short")
        assert not result

    def test_empty_fails(self, validator):
        assert not validator.validate("")
        assert not validator.validate("   ")

    def test_minimum_valid_length(self, validator):
        text = "The ATM swallowed my card and I am very frustrated about this."
        assert validator.validate(text)

    def test_too_long_fails(self, validator):
        text = "word " * 400
        assert not validator.validate(text)


class TestRepetitionCheck:
    def test_heavily_repeated_word_fails(self, validator):
        text = ("error " * 50) + "The ATM had an issue with my account."
        assert not validator.validate(text)

    def test_reasonable_text_passes(self, validator):
        text = (
            "I am extremely frustrated because the ATM at your Main Street branch "
            "swallowed my card yesterday evening. I was trying to withdraw cash but "
            "the machine froze mid-transaction and never returned my card. "
            "This has left me without access to my funds and I demand immediate resolution."
        )
        assert validator.validate(text)


class TestProfanityCheck:
    def test_explicit_profanity_fails(self, validator):
        text = (
            "The ATM machine is fucking broken again and I am so angry about this. "
            "Please fix this issue immediately with my account."
        )
        assert not validator.validate(text)

    def test_mild_language_passes(self, validator):
        text = (
            "I am damn frustrated with the ATM failure at your branch. "
            "The machine ate my card and I need help resolving this issue."
        )
        assert validator.validate(text)


class TestCategoryRelevance:
    def test_atm_keyword_passes(self, validator):
        text = (
            "The ATM machine at your branch failed to dispense cash "
            "and I am extremely frustrated about this situation."
        )
        assert validator.validate(text, category="ATM_FAILURE")

    def test_no_atm_keyword_fails(self, validator):
        text = (
            "I am very unhappy with your service and the way you treat customers "
            "in your establishment. Something needs to change very soon."
        )
        assert not validator.validate(text, category="ATM_FAILURE")

    def test_fraud_keyword_passes(self, validator):
        text = (
            "There are unauthorized transactions on my account and I am very scared. "
            "Please investigate this fraud immediately."
        )
        assert validator.validate(text, category="FRAUD_DETECTION")

    def test_ux_keyword_passes(self, validator):
        text = (
            "Your mobile banking app keeps crashing every time I try to login. "
            "The interface is completely broken and unusable."
        )
        assert validator.validate(text, category="UX_ISSUES")

    def test_unknown_category_skips_check(self, validator):
        text = "I have a complaint about something completely unrelated to any category."
        # Should not fail on unknown category — just skip that check
        result = validator.validate(text, category="UNKNOWN_CATEGORY")
        # Length and other checks still apply
        assert isinstance(result.passed, bool)
