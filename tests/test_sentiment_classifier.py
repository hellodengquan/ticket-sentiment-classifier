import pytest
from ticket_classifier.sentiment_classifier import SentimentClassifier


class TestSentimentClassifier:
    @pytest.fixture
    def classifier(self):
        return SentimentClassifier(random_state=42)

    @pytest.fixture
    def trained_classifier(self, classifier):
        texts = [
            "This product is amazing, I love it!",
            "Great service, highly recommended!",
            "Excellent quality, worth every penny.",
            "The best experience I've had so far.",
            "Super happy with my purchase.",
            "This is terrible, very disappointed.",
            "Worst customer service ever.",
            "Horrible quality, don't buy it.",
            "Completely useless, waste of money.",
            "I hate this product so much.",
            "The product is okay, nothing special.",
            "It works as expected, no complaints.",
            "Average quality, could be better.",
            "Not bad, but not great either.",
            "It's fine, meets basic needs.",
        ]
        labels = [
            "positive", "positive", "positive", "positive", "positive",
            "negative", "negative", "negative", "negative", "negative",
            "neutral", "neutral", "neutral", "neutral", "neutral"
        ]
        classifier.train(texts, labels)
        return classifier

    def test_init(self, classifier):
        assert not classifier.is_trained
        assert classifier.model is not None

    def test_train_success(self, classifier):
        texts = [
            "Great product!", "Love it!", "Amazing quality!", "Best ever!", "Happy!",
            "Terrible!", "Worst!", "Horrible!", "Useless!", "Hate it!",
            "Okay.", "Fine.", "Average.", "Not bad.", "Neutral."
        ] * 2
        labels = [
            "positive"] * 10 + ["negative"] * 10 + ["neutral"] * 10
        result = classifier.train(texts, labels)
        assert classifier.is_trained
        assert "accuracy" in result
        assert result["train_size"] > 0
        assert result["test_size"] > 0

    def test_train_mismatched_lengths(self, classifier):
        with pytest.raises(ValueError, match="Number of texts and labels must match"):
            classifier.train(["text1", "text2"], ["positive"])

    def test_train_insufficient_samples(self, classifier):
        texts = ["text1", "text2"]
        labels = ["positive", "negative"]
        with pytest.raises(ValueError, match="Need at least 10 samples"):
            classifier.train(texts, labels)

    def test_train_invalid_label(self, classifier):
        texts = ["text1"] * 15
        labels = ["positive"] * 5 + ["negative"] * 5 + ["invalid"] * 5
        with pytest.raises(ValueError, match="Invalid label"):
            classifier.train(texts, labels)

    def test_predict_positive(self, trained_classifier):
        result = trained_classifier.predict("This product is absolutely wonderful!")
        assert result["sentiment"] in trained_classifier.SENTIMENT_LABELS
        assert 0 <= result["confidence"] <= 1
        assert "probabilities" in result
        for label in trained_classifier.SENTIMENT_LABELS:
            assert label in result["probabilities"]

    def test_predict_negative(self, trained_classifier):
        result = trained_classifier.predict("This is the worst product ever, I hate it!")
        assert result["sentiment"] in trained_classifier.SENTIMENT_LABELS

    def test_predict_neutral(self, trained_classifier):
        result = trained_classifier.predict("The product is okay, works as expected.")
        assert result["sentiment"] in trained_classifier.SENTIMENT_LABELS

    def test_predict_without_training(self, classifier):
        with pytest.raises(RuntimeError, match="Model is not trained"):
            classifier.predict("test text")

    def test_predict_batch(self, trained_classifier):
        texts = [
            "This is amazing!",
            "This is terrible!",
            "This is okay."
        ]
        results = trained_classifier.predict_batch(texts)
        assert len(results) == 3
        for r in results:
            assert "sentiment" in r
            assert "confidence" in r

    def test_get_low_confidence_samples(self, trained_classifier):
        texts = [
            "This is a very ambiguous statement that could go either way really.",
            "Amazing product love it!",
            "Terrible hate it worst ever!"
        ]
        low_conf = trained_classifier.get_low_confidence_samples(texts, threshold=0.9)
        assert isinstance(low_conf, list)
        for sample in low_conf:
            assert "index" in sample
            assert "text" in sample
            assert "confidence" in sample
            assert sample["confidence"] < 0.9

    def test_sentiment_labels_constant(self):
        labels = SentimentClassifier.SENTIMENT_LABELS
        assert "positive" in labels
        assert "negative" in labels
        assert "neutral" in labels
