import pytest
import json
import tempfile
import os
from pathlib import Path
from ticket_classifier.report_generator import ReportGenerator


class TestReportGenerator:
    @pytest.fixture
    def report_gen(self):
        return ReportGenerator(low_confidence_threshold=0.7)

    @pytest.fixture
    def sample_predictions(self):
        return [
            {"sentiment": "positive", "confidence": 0.95, "probabilities": {"positive": 0.95, "neutral": 0.03, "negative": 0.02}},
            {"sentiment": "positive", "confidence": 0.88, "probabilities": {"positive": 0.88, "neutral": 0.08, "negative": 0.04}},
            {"sentiment": "negative", "confidence": 0.92, "probabilities": {"negative": 0.92, "neutral": 0.05, "positive": 0.03}},
            {"sentiment": "neutral", "confidence": 0.55, "probabilities": {"neutral": 0.55, "positive": 0.30, "negative": 0.15}},
            {"sentiment": "positive", "confidence": 0.60, "probabilities": {"positive": 0.60, "neutral": 0.25, "negative": 0.15}},
        ]

    @pytest.fixture
    def sample_tickets(self):
        return [
            {"id": 0, "file_name": "ticket_0.txt", "content": "Great product!"},
            {"id": 1, "file_name": "ticket_1.txt", "content": "Love it!"},
            {"id": 2, "file_name": "ticket_2.txt", "content": "Terrible service!"},
            {"id": 3, "file_name": "ticket_3.txt", "content": "It's okay I guess."},
            {"id": 4, "file_name": "ticket_4.txt", "content": "Not sure how I feel."},
        ]

    def test_init(self, report_gen):
        assert report_gen.low_confidence_threshold == 0.7

    def test_generate_batch_summary(self, report_gen, sample_predictions, sample_tickets):
        summary = report_gen.generate_batch_summary(sample_predictions, sample_tickets)

        assert summary["total_tickets"] == 5
        assert "sentiment_distribution" in summary
        assert "sentiment_percentages" in summary
        assert "confidence_stats" in summary
        assert summary["low_confidence_count"] == 2
        assert summary["low_confidence_percentage"] == 40.0

        assert "average" in summary["confidence_stats"]
        assert "min" in summary["confidence_stats"]
        assert "max" in summary["confidence_stats"]

        assert len(summary["low_confidence_samples"]) == 2
        assert summary["low_confidence_samples"][0]["file_name"] == "ticket_3.txt"

    def test_generate_batch_summary_without_tickets(self, report_gen, sample_predictions):
        summary = report_gen.generate_batch_summary(sample_predictions)

        assert summary["total_tickets"] == 5
        assert len(summary["low_confidence_samples"]) == 2
        assert summary["low_confidence_samples"][0]["file_name"] == "ticket_3"

    def test_generate_batch_summary_empty_predictions(self, report_gen):
        with pytest.raises(ValueError, match="No predictions provided"):
            report_gen.generate_batch_summary([])

    def test_generate_low_confidence_report_json(self, report_gen, sample_predictions):
        low_conf_samples = [
            {
                "index": 3,
                "file_name": "ticket_3.txt",
                "sentiment": "neutral",
                "confidence": 0.55,
                "probabilities": {"neutral": 0.55, "positive": 0.30, "negative": 0.15}
            }
        ]

        report_json = report_gen.generate_low_confidence_report(low_conf_samples)
        report_data = json.loads(report_json)

        assert report_data["report_type"] == "low_confidence_tickets"
        assert report_data["threshold"] == 0.7
        assert report_data["total_samples"] == 1
        assert len(report_data["samples"]) == 1

    def test_generate_low_confidence_report_to_file(self, report_gen):
        low_conf_samples = [
            {
                "index": 0,
                "file_name": "ticket_0.txt",
                "sentiment": "neutral",
                "confidence": 0.5,
                "probabilities": {}
            }
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            result_path = report_gen.generate_low_confidence_report(low_conf_samples, output_path)
            assert result_path == output_path
            assert Path(output_path).exists()

            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert data["total_samples"] == 1
        finally:
            os.unlink(output_path)

    def test_generate_full_report_json(self, report_gen, sample_predictions, sample_tickets):
        outputs = report_gen.generate_full_report(sample_predictions, sample_tickets)

        assert "batch_summary" in outputs
        assert "low_confidence_report" in outputs
        assert "predictions_csv" in outputs

        summary_data = json.loads(outputs["batch_summary"])
        assert summary_data["total_tickets"] == 5

        low_conf_data = json.loads(outputs["low_confidence_report"])
        assert low_conf_data["total_samples"] == 2

        assert "sentiment" in outputs["predictions_csv"]

    def test_generate_full_report_to_dir(self, report_gen, sample_predictions, sample_tickets):
        with tempfile.TemporaryDirectory() as temp_dir:
            outputs = report_gen.generate_full_report(sample_predictions, sample_tickets, temp_dir)

            assert Path(outputs["batch_summary"]).exists()
            assert Path(outputs["low_confidence_report"]).exists()
            assert Path(outputs["predictions_csv"]).exists()

            with open(outputs["batch_summary"], "r", encoding="utf-8") as f:
                summary = json.load(f)
            assert summary["total_tickets"] == 5

    def test_generate_full_report_creates_dir(self, report_gen, sample_predictions):
        base_dir = tempfile.gettempdir()
        output_dir = os.path.join(base_dir, "test_reports_" + str(os.getpid()))

        try:
            outputs = report_gen.generate_full_report(sample_predictions, output_dir=output_dir)
            assert Path(output_dir).exists()
        finally:
            if Path(output_dir).exists():
                for f in Path(output_dir).glob("*"):
                    f.unlink()
                Path(output_dir).rmdir()

    def test_custom_threshold(self):
        report_gen = ReportGenerator(low_confidence_threshold=0.5)
        predictions = [
            {"sentiment": "positive", "confidence": 0.6, "probabilities": {}},
            {"sentiment": "negative", "confidence": 0.4, "probabilities": {}},
        ]
        summary = report_gen.generate_batch_summary(predictions)
        assert summary["low_confidence_count"] == 1
