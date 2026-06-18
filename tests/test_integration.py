import pytest
import tempfile
import json
from pathlib import Path
from ticket_classifier import TextReader, SentimentClassifier, ReportGenerator


class TestIntegration:
    @pytest.fixture
    def trained_classifier(self):
        classifier = SentimentClassifier(random_state=42)
        texts = [
            "This product is amazing, I love it!",
            "Great service, highly recommended!",
            "Excellent quality, worth every penny.",
            "The best experience I've had so far.",
            "Super happy with my purchase.",
            "Wonderful product, fantastic experience!",
            "Love everything about this, perfect!",
            "Amazing customer support, thank you!",
            "Incredible value for money, buy it!",
            "Absolutely fantastic, 5 stars!",
            "This is terrible, very disappointed.",
            "Worst customer service ever.",
            "Horrible quality, don't buy it.",
            "Completely useless, waste of money.",
            "I hate this product so much.",
            "Awful experience, never coming back.",
            "Broken on arrival, total garbage.",
            "Terrible, would not recommend.",
            "Disgusting quality, avoid at all costs.",
            "Worst purchase ever made.",
            "The product is okay, nothing special.",
            "It works as expected, no complaints.",
            "Average quality, could be better.",
            "Not bad, but not great either.",
            "It's fine, meets basic needs.",
            "Decent product, nothing extraordinary.",
            "Mediocre at best, pretty standard.",
            "Acceptable, but I've seen better.",
            "Fair quality, nothing to write home about.",
            "It's alright, does the job.",
        ]
        labels = (
            ["positive"] * 10
            + ["negative"] * 10
            + ["neutral"] * 10
        )
        classifier.train(texts, labels)
        return classifier

    @pytest.fixture
    def sample_tickets_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tickets = [
                ("ticket_001.txt", "This product is absolutely wonderful! Best purchase ever."),
                ("ticket_002.txt", "Terrible experience, the product broke immediately."),
                ("ticket_003.txt", "The product is okay, works as advertised."),
                ("ticket_004.txt", "Love it! Great quality and fast shipping."),
                ("ticket_005.txt", "This is somewhat ambiguous, could be better or worse."),
            ]
            for filename, content in tickets:
                Path(temp_dir) / filename
                (Path(temp_dir) / filename).write_text(content, encoding="utf-8")
            yield temp_dir

    def test_full_pipeline(self, trained_classifier, sample_tickets_dir):
        reader = TextReader()
        report_gen = ReportGenerator(low_confidence_threshold=0.7)

        tickets = reader.read_directory(sample_tickets_dir)
        assert len(tickets) == 5

        texts = [t["content"] for t in tickets]
        predictions = trained_classifier.predict_batch(texts)
        assert len(predictions) == 5

        for pred in predictions:
            assert "sentiment" in pred
            assert "confidence" in pred
            assert "probabilities" in pred

        summary = report_gen.generate_batch_summary(predictions, tickets)
        assert summary["total_tickets"] == 5
        assert "positive" in summary["sentiment_distribution"]
        assert summary["low_confidence_count"] >= 0

    def test_full_report_generation(self, trained_classifier, sample_tickets_dir):
        reader = TextReader()
        report_gen = ReportGenerator(low_confidence_threshold=0.7)

        tickets = reader.read_directory(sample_tickets_dir)
        texts = [t["content"] for t in tickets]
        predictions = trained_classifier.predict_batch(texts)

        with tempfile.TemporaryDirectory() as output_dir:
            outputs = report_gen.generate_full_report(predictions, tickets, output_dir)

            assert Path(outputs["batch_summary"]).exists()
            assert Path(outputs["low_confidence_report"]).exists()
            assert Path(outputs["predictions_csv"]).exists()

            with open(outputs["batch_summary"], "r", encoding="utf-8") as f:
                summary = json.load(f)
            assert summary["total_tickets"] == 5

            with open(outputs["low_confidence_report"], "r", encoding="utf-8") as f:
                low_conf = json.load(f)
            assert "samples" in low_conf

    def test_low_confidence_detection(self, trained_classifier):
        ambiguous_texts = [
            "This could be better but also could be worse, not sure really.",
            "I have mixed feelings about this purchase.",
            "It's complicated, there are pros and cons.",
        ]
        low_conf = trained_classifier.get_low_confidence_samples(
            ambiguous_texts, threshold=0.8
        )
        assert len(low_conf) >= 0
        for sample in low_conf:
            assert sample["confidence"] < 0.8

    def test_sentiment_distribution(self, trained_classifier):
        texts = [
            "Amazing product love it!",
            "Terrible worst ever hate it!",
            "It's okay nothing special.",
            "Great quality highly recommend!",
            "Horrible experience never again!",
        ]
        predictions = trained_classifier.predict_batch(texts)
        sentiments = [p["sentiment"] for p in predictions]
        assert "positive" in sentiments
        assert "negative" in sentiments
