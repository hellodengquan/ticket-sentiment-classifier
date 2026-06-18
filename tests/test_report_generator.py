import pytest
import json
import csv
import io
import tempfile
import os
from pathlib import Path
from ticket_classifier.report_generator import ReportGenerator, VALID_OUTPUT_FORMATS


class TestReportGenerator:
    @pytest.fixture
    def report_gen(self):
        return ReportGenerator(low_confidence_threshold=0.7, output_format="both")

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
        assert report_gen.output_format == "both"

    def test_init_invalid_output_format(self):
        with pytest.raises(ValueError, match="output_format must be one of"):
            ReportGenerator(output_format="xml")

    def test_valid_output_formats(self):
        assert VALID_OUTPUT_FORMATS == {"json", "csv", "both"}

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

    def test_generate_low_confidence_report_json(self, report_gen):
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

    def test_custom_threshold(self):
        report_gen = ReportGenerator(low_confidence_threshold=0.5)
        predictions = [
            {"sentiment": "positive", "confidence": 0.6, "probabilities": {}},
            {"sentiment": "negative", "confidence": 0.4, "probabilities": {}},
        ]
        summary = report_gen.generate_batch_summary(predictions)
        assert summary["low_confidence_count"] == 1


class TestReportGeneratorOutputFormat:
    @pytest.fixture
    def sample_predictions(self):
        return [
            {"sentiment": "positive", "confidence": 0.95, "probabilities": {"positive": 0.95, "neutral": 0.03, "negative": 0.02}},
            {"sentiment": "negative", "confidence": 0.55, "probabilities": {"negative": 0.55, "neutral": 0.30, "positive": 0.15}},
        ]

    @pytest.fixture
    def sample_tickets(self):
        return [
            {"id": 0, "file_name": "ticket_0.txt", "content": "Great!"},
            {"id": 1, "file_name": "ticket_1.txt", "content": "Not sure."},
        ]

    def test_json_only_in_memory(self, sample_predictions, sample_tickets):
        gen = ReportGenerator(output_format="json")
        outputs = gen.generate_full_report(sample_predictions, sample_tickets)

        assert "batch_summary" in outputs
        assert "low_confidence_report" in outputs
        assert "predictions_csv" not in outputs
        assert "low_confidence_csv" not in outputs

        summary = json.loads(outputs["batch_summary"])
        assert summary["total_tickets"] == 2

    def test_csv_only_in_memory(self, sample_predictions, sample_tickets):
        gen = ReportGenerator(output_format="csv")
        outputs = gen.generate_full_report(sample_predictions, sample_tickets)

        assert "predictions_csv" in outputs
        assert "low_confidence_csv" in outputs
        assert "batch_summary" not in outputs
        assert "low_confidence_report" not in outputs

        reader = csv.DictReader(io.StringIO(outputs["predictions_csv"]))
        rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["sentiment"] == "positive"

    def test_both_in_memory(self, sample_predictions, sample_tickets):
        gen = ReportGenerator(output_format="both")
        outputs = gen.generate_full_report(sample_predictions, sample_tickets)

        assert "batch_summary" in outputs
        assert "low_confidence_report" in outputs
        assert "predictions_csv" in outputs
        assert "low_confidence_csv" in outputs

    def test_json_only_to_dir(self, sample_predictions, sample_tickets):
        gen = ReportGenerator(output_format="json")
        with tempfile.TemporaryDirectory() as temp_dir:
            outputs = gen.generate_full_report(sample_predictions, sample_tickets, output_dir=temp_dir)

            assert "batch_summary" in outputs
            assert "low_confidence_report" in outputs
            assert "predictions_csv" not in outputs
            assert "low_confidence_csv" not in outputs

            assert Path(outputs["batch_summary"]).exists()
            assert Path(outputs["low_confidence_report"]).exists()
            assert not (Path(temp_dir) / "predictions.csv").exists()

    def test_csv_only_to_dir(self, sample_predictions, sample_tickets):
        gen = ReportGenerator(output_format="csv")
        with tempfile.TemporaryDirectory() as temp_dir:
            outputs = gen.generate_full_report(sample_predictions, sample_tickets, output_dir=temp_dir)

            assert "predictions_csv" in outputs
            assert "low_confidence_csv" in outputs
            assert "batch_summary" not in outputs
            assert "low_confidence_report" not in outputs

            assert Path(outputs["predictions_csv"]).exists()
            assert Path(outputs["low_confidence_csv"]).exists()
            assert not (Path(temp_dir) / "batch_summary.json").exists()

    def test_both_to_dir(self, sample_predictions, sample_tickets):
        gen = ReportGenerator(output_format="both")
        with tempfile.TemporaryDirectory() as temp_dir:
            outputs = gen.generate_full_report(sample_predictions, sample_tickets, output_dir=temp_dir)

            assert Path(outputs["batch_summary"]).exists()
            assert Path(outputs["low_confidence_report"]).exists()
            assert Path(outputs["predictions_csv"]).exists()
            assert Path(outputs["low_confidence_csv"]).exists()

    def test_per_call_format_override(self, sample_predictions, sample_tickets):
        gen = ReportGenerator(output_format="both")
        with tempfile.TemporaryDirectory() as temp_dir:
            outputs = gen.generate_full_report(
                sample_predictions, sample_tickets,
                output_dir=temp_dir,
                output_format="json"
            )
            assert "batch_summary" in outputs
            assert "predictions_csv" not in outputs

    def test_per_call_format_override_csv(self, sample_predictions, sample_tickets):
        gen = ReportGenerator(output_format="json")
        outputs = gen.generate_full_report(
            sample_predictions, sample_tickets,
            output_format="csv"
        )
        assert "predictions_csv" in outputs
        assert "batch_summary" not in outputs

    def test_per_call_invalid_format(self, sample_predictions, sample_tickets):
        gen = ReportGenerator(output_format="both")
        with pytest.raises(ValueError, match="output_format must be one of"):
            gen.generate_full_report(
                sample_predictions, sample_tickets,
                output_format="xml"
            )

    def test_low_confidence_csv_content(self, sample_predictions, sample_tickets):
        gen = ReportGenerator(low_confidence_threshold=0.7, output_format="csv")
        outputs = gen.generate_full_report(sample_predictions, sample_tickets)

        reader = csv.DictReader(io.StringIO(outputs["low_confidence_csv"]))
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["file_name"] == "ticket_1.txt"
        assert float(rows[0]["confidence"]) < 0.7

    def test_predictions_csv_has_probabilities(self, sample_predictions, sample_tickets):
        gen = ReportGenerator(output_format="csv")
        outputs = gen.generate_full_report(sample_predictions, sample_tickets)

        reader = csv.DictReader(io.StringIO(outputs["predictions_csv"]))
        rows = list(reader)
        assert "prob_negative" in rows[0]
        assert "prob_neutral" in rows[0]
        assert "prob_positive" in rows[0]

    def test_low_confidence_csv_has_probabilities(self, sample_predictions, sample_tickets):
        gen = ReportGenerator(low_confidence_threshold=0.7, output_format="csv")
        outputs = gen.generate_full_report(sample_predictions, sample_tickets)

        reader = csv.DictReader(io.StringIO(outputs["low_confidence_csv"]))
        rows = list(reader)
        assert len(rows) == 1
        assert "prob_negative" in rows[0]
        assert "prob_neutral" in rows[0]
        assert "prob_positive" in rows[0]

    def test_empty_low_confidence_csv(self):
        gen = ReportGenerator(low_confidence_threshold=0.3, output_format="csv")
        predictions = [
            {"sentiment": "positive", "confidence": 0.95, "probabilities": {"positive": 0.95, "neutral": 0.03, "negative": 0.02}},
        ]
        outputs = gen.generate_full_report(predictions)
        reader = csv.DictReader(io.StringIO(outputs["low_confidence_csv"]))
        rows = list(reader)
        assert len(rows) == 0
