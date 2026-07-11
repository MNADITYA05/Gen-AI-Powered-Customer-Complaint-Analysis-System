"""Unit tests for ComplaintClassifier."""
import pytest

from core.analysis.classifier import ComplaintClassifier


@pytest.fixture(scope="module")
def trained_classifier(sample_complaints_df):
    clf = ComplaintClassifier(model_dir="/tmp/test_models_clf")
    clf.train(sample_complaints_df)
    return clf


class TestUntrained:
    def test_predict_raises_when_not_trained(self):
        clf = ComplaintClassifier()
        with pytest.raises(RuntimeError):
            clf.predict("test text")

    def test_get_info_reports_not_trained(self):
        clf = ComplaintClassifier()
        assert clf.get_info()["is_trained"] is False


class TestTrained:
    def test_is_trained_after_train(self, trained_classifier):
        assert trained_classifier.is_trained is True

    def test_predict_returns_required_keys(self, trained_classifier):
        result = trained_classifier.predict("The ATM swallowed my card and I am furious")
        assert {"category", "emotion", "severity", "confidence"}.issubset(result.keys())

    def test_confidence_values_between_0_and_1(self, trained_classifier):
        result = trained_classifier.predict("Unauthorized transaction on my account")
        for v in result["confidence"].values():
            assert 0.0 <= v <= 1.0

    def test_predict_batch(self, trained_classifier):
        texts = [
            "The ATM machine failed to return my card",
            "App keeps crashing when I try to login",
        ]
        results = trained_classifier.predict_batch(texts)
        assert len(results) == 2
        for r in results:
            assert "category" in r

    def test_get_info_has_classes(self, trained_classifier):
        info = trained_classifier.get_info()
        assert len(info["category_classes"]) > 0
        assert len(info["emotion_classes"]) > 0
        assert len(info["severity_classes"]) > 0

    def test_save_and_reload(self, trained_classifier, sample_complaints_df, tmp_path):
        trained_classifier._model_dir = tmp_path
        trained_classifier.save()

        fresh = ComplaintClassifier(model_dir=str(tmp_path))
        loaded = fresh.load()
        assert loaded is True
        assert fresh.is_trained is True

        result = fresh.predict("ATM did not dispense cash after deducting from account")
        assert result["category"] in ["ATM_FAILURE", "FRAUD_DETECTION", "UX_ISSUES"]
