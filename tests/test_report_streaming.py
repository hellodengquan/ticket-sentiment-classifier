import pytest
import csv
import io
import tempfile
import os
from pathlib import Path
from ticket_classifier.report_generator import ReportGenerator


def make_predictions(n: int, seed: int = 0):
    labels = ["positive", "neutral", "negative"]
    probs_list = [
        {"positive": 0.95, "neutral": 0.03, "negative": 0.02},
        {"neutral": 0.55, "positive": 0.30, "negative": 0.15},
        {"negative": 0.90, "neutral": 0.07, "positive": 0.03},
        {"positive": 0.60, "neutral": 0.25, "negative": 0.15},
    ]
    confs = [0.95, 0.55, 0.90, 0.60]
    result = []
    for i in range(n):
        j = i % len(probs_list)
        result.append({
            "sentiment": labels[j % 3],
            "confidence": confs[j],
            "probabilities": dict(probs_list[j])
        })
    return result


def make_tickets(n: int):
    return [
        {"id": i, "file_name": f"ticket_{i:06d}.txt", "content": f"Content for ticket {i}"}
        for i in range(n)
    ]


class TestStreamingCSV:
    @pytest.fixture
    def gen(self):
        return ReportGenerator(low_confidence_threshold=0.7)

    def test_stream_predictions_csv_to_memory(self, gen):
        preds = make_predictions(5)
        tickets = make_tickets(5)
        csv_str = gen.stream_predictions_csv(iter(preds), tickets)
        assert isinstance(csv_str, str)
        reader = csv.DictReader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == 5
        assert rows[0]["sentiment"] == "positive"
        assert rows[0]["file_name"] == "ticket_000000.txt"

    def test_stream_predictions_csv_to_file(self, gen):
        preds = make_predictions(10)
        tickets = make_tickets(10)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_path = f.name
        try:
            result = gen.stream_predictions_csv(iter(preds), tickets, temp_path)
            assert os.path.exists(temp_path)
            assert isinstance(result, str)
            with open(temp_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert content == result
            reader = csv.DictReader(io.StringIO(content))
            rows = list(reader)
            assert len(rows) == 10
        finally:
            os.unlink(temp_path)

    def test_stream_low_confidence_csv_to_memory(self, gen):
        preds = make_predictions(20)
        low_conf = list(gen.iter_low_confidence_samples(iter(preds), make_tickets(20)))
        csv_str = gen.stream_low_confidence_csv(iter(low_conf))
        reader = csv.DictReader(io.StringIO(csv_str))
        rows = list(reader)
        for row in rows:
            assert float(row["confidence"]) < 0.7

    def test_stream_low_confidence_csv_to_file(self, gen):
        preds = make_predictions(20)
        low_conf = list(gen.iter_low_confidence_samples(iter(preds), make_tickets(20)))
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_path = f.name
        try:
            result = gen.stream_low_confidence_csv(iter(low_conf), temp_path)
            assert os.path.exists(temp_path)
            assert isinstance(result, str)
            with open(temp_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert content == result
            reader = csv.DictReader(io.StringIO(content))
            rows = list(reader)
            for row in rows:
                assert float(row["confidence"]) < 0.7
        finally:
            os.unlink(temp_path)

    def test_stream_large_predictions_no_memory_leak(self, gen):
        n = 5000
        preds = make_predictions(n)
        tickets = make_tickets(n)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_path = f.name
        try:
            content = gen.stream_predictions_csv(iter(preds), tickets, temp_path)
            assert os.path.exists(temp_path)
            assert isinstance(content, str)
            reader = csv.DictReader(io.StringIO(content))
            rows = list(reader)
            assert len(rows) == n
            assert "prob_positive" in rows[0]
            assert "prob_neutral" in rows[0]
            assert "prob_negative" in rows[0]
        finally:
            os.unlink(temp_path)

    def test_stream_large_low_confidence(self, gen):
        n = 5000
        preds = make_predictions(n)
        tickets = make_tickets(n)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_path = f.name
        try:
            low_conf_iter = gen.iter_low_confidence_samples(iter(preds), tickets)
            content = gen.stream_low_confidence_csv(low_conf_iter, temp_path)
            assert os.path.exists(temp_path)
            reader = csv.DictReader(io.StringIO(content))
            rows = list(reader)
            for row in rows:
                assert float(row["confidence"]) < 0.7
            assert len(rows) > 0
        finally:
            os.unlink(temp_path)

    def test_iter_predictions_csv_rows(self, gen):
        preds = make_predictions(20)
        tickets = make_tickets(20)
        chunks = list(gen.iter_predictions_csv_rows(iter(preds), tickets))
        assert len(chunks) >= 20
        combined = "".join(chunks)
        reader = csv.DictReader(io.StringIO(combined))
        rows = list(reader)
        assert len(rows) == 20

    def test_iter_predictions_csv_rows_consumes_generator(self, gen):
        n = 100
        preds = (p for p in make_predictions(n))
        tickets = make_tickets(n)
        chunks = list(gen.iter_predictions_csv_rows(preds, tickets))
        combined = "".join(chunks)
        reader = csv.DictReader(io.StringIO(combined))
        rows = list(reader)
        assert len(rows) == n

    def test_streaming_produces_same_result_as_batched(self, gen):
        preds = make_predictions(100)
        tickets = make_tickets(100)

        in_memory_csv = gen.stream_predictions_csv(iter(preds), tickets)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_path = f.name
        try:
            gen.stream_predictions_csv(iter(preds), tickets, temp_path)
            with open(temp_path, "r", encoding="utf-8") as f:
                streamed = f.read()
            assert streamed == in_memory_csv
        finally:
            os.unlink(temp_path)

    def test_stream_predictions_without_tickets(self, gen):
        preds = make_predictions(5)
        csv_str = gen.stream_predictions_csv(iter(preds))
        reader = csv.DictReader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == 5
        assert rows[0]["file_name"] == "ticket_0"

    def test_stream_empty_predictions(self, gen):
        result = gen.stream_predictions_csv(iter([]))
        assert result is not None
        assert "id,file_name,sentiment,confidence" in result

    def test_stream_empty_low_confidence(self, gen):
        result = gen.stream_low_confidence_csv(iter([]))
        assert "index,file_name,sentiment,confidence" in result


class TestIterLowConfidence:
    @pytest.fixture
    def gen(self):
        return ReportGenerator(low_confidence_threshold=0.7)

    def test_iter_low_confidence_basic(self, gen):
        preds = make_predictions(20)
        tickets = make_tickets(20)
        low_confs = list(gen.iter_low_confidence_samples(iter(preds), tickets))
        for sample in low_confs:
            assert sample["confidence"] < 0.7
            assert "index" in sample
            assert "file_name" in sample
            assert "probabilities" in sample

    def test_iter_low_confidence_lazy(self, gen):
        n = 100
        consumed = [0]

        def gen_preds():
            for p in make_predictions(n):
                consumed[0] += 1
                yield p

        tickets = make_tickets(n)
        iter_obj = gen.iter_low_confidence_samples(gen_preds(), tickets)
        assert consumed[0] == 0
        first = next(iter_obj)
        assert consumed[0] >= 1
        assert first["confidence"] < 0.7

    def test_iter_low_confidence_without_tickets(self, gen):
        preds = make_predictions(10)
        low_confs = list(gen.iter_low_confidence_samples(iter(preds)))
        for sample in low_confs:
            assert sample["file_name"].startswith("ticket_")

    def test_iter_low_confidence_none_below_threshold(self):
        gen = ReportGenerator(low_confidence_threshold=0.01)
        preds = make_predictions(20)
        low_confs = list(gen.iter_low_confidence_samples(iter(preds)))
        assert len(low_confs) == 0

    def test_iter_low_confidence_all_below_threshold(self):
        gen = ReportGenerator(low_confidence_threshold=0.99)
        preds = make_predictions(10)
        low_confs = list(gen.iter_low_confidence_samples(iter(preds)))
        assert len(low_confs) == 10

    def test_iter_low_confidence_handles_large_generator(self, gen):
        n = 10000
        preds = (p for p in make_predictions(n))
        tickets = make_tickets(n)
        count = 0
        for sample in gen.iter_low_confidence_samples(preds, tickets):
            assert sample["confidence"] < 0.7
            count += 1
        assert count > 0
        assert count < n


class TestStreamingIntegration:
    def test_full_report_uses_streaming_csv(self):
        gen = ReportGenerator(low_confidence_threshold=0.7, output_format="csv")
        n = 1000
        preds = make_predictions(n)
        tickets = make_tickets(n)
        with tempfile.TemporaryDirectory() as temp_dir:
            outputs = gen.generate_full_report(preds, tickets, output_dir=temp_dir)
            assert "predictions_csv" in outputs
            assert "low_confidence_csv" in outputs

            with open(outputs["predictions_csv"], "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == n

            with open(outputs["low_confidence_csv"], "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            for row in rows:
                assert float(row["confidence"]) < 0.7

    def test_streaming_consistent_with_batch_api(self):
        gen = ReportGenerator(low_confidence_threshold=0.7, output_format="csv")
        preds = make_predictions(50)
        tickets = make_tickets(50)
        batch_out = gen.generate_full_report(preds, tickets)
        stream_preds_csv = gen.stream_predictions_csv(iter(preds), tickets)
        assert batch_out["predictions_csv"] == stream_preds_csv

    def test_streaming_correct_row_count_batched_vs_generator(self):
        n = 2000
        gen = ReportGenerator(low_confidence_threshold=0.7)
        preds = make_predictions(n)
        tickets = make_tickets(n)

        csv_str_list = gen.stream_predictions_csv(iter(preds), tickets)
        rows_a = len(list(csv.DictReader(io.StringIO(csv_str_list))))

        csv_str_gen = gen.stream_predictions_csv((p for p in preds), tickets)
        rows_b = len(list(csv.DictReader(io.StringIO(csv_str_gen))))

        assert rows_a == rows_b == n
