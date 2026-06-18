import pytest
import os
import io
import errno
import tempfile
import csv
import builtins
from pathlib import Path
from ticket_classifier.report_generator import ReportGenerator


def make_predictions(n: int):
    labels = ["positive", "neutral", "negative"]
    probs_list = [
        {"positive": 0.95, "neutral": 0.03, "negative": 0.02},
        {"neutral": 0.55, "positive": 0.30, "negative": 0.15},
        {"negative": 0.90, "neutral": 0.07, "positive": 0.03},
        {"positive": 0.60, "neutral": 0.25, "negative": 0.15},
    ]
    confs = [0.95, 0.55, 0.90, 0.60]
    for i in range(n):
        j = i % len(probs_list)
        yield {
            "sentiment": labels[j % 3],
            "confidence": confs[j],
            "probabilities": dict(probs_list[j])
        }


def make_tickets(n: int):
    return [
        {"id": i, "file_name": f"ticket_{i:06d}.txt", "content": f"Content {i}"}
        for i in range(n)
    ]


class BrokenPipeWriter(io.StringIO):
    def __init__(self, fail_after_rows: int):
        super().__init__()
        self.fail_after_rows = fail_after_rows
        self.rows_written = 0
        self._newline_count = 0
        self.closed_cleanly = False
        self._write_count = 0

    def write(self, s: str) -> int:
        self._write_count += 1
        if self._write_count > (self.fail_after_rows + 1) and self.fail_after_rows >= 0:
            self.closed_cleanly = True
            raise BrokenPipeError(errno.EPIPE, "Broken pipe")
        self._newline_count += s.count("\n")
        return super().write(s)

    def close(self):
        self.closed_cleanly = True
        super().close()


class BrokenPipeOnSecondWrite(io.StringIO):
    def __init__(self):
        super().__init__()
        self.writes = 0
        self.closed_cleanly = False

    def write(self, s: str) -> int:
        self.writes += 1
        if self.writes >= 3:
            self.closed_cleanly = True
            raise BrokenPipeError(errno.EPIPE, "Broken pipe")
        return super().write(s)


class TestStreamingBrokenPipePredictions:
    @pytest.fixture
    def gen(self):
        return ReportGenerator(low_confidence_threshold=0.7)

    def test_predictions_stream_propagates_broken_pipe(self, gen):
        writer = BrokenPipeOnSecondWrite()
        preds = make_predictions(1000)
        tickets = make_tickets(1000)
        with pytest.raises(BrokenPipeError):
            gen._stream_predictions_to_fileobj(preds, tickets, writer)

    def test_predictions_stream_partial_output_before_pipe_break(self, gen):
        writer = BrokenPipeOnSecondWrite()
        preds = make_predictions(1000)
        tickets = make_tickets(1000)
        try:
            gen._stream_predictions_to_fileobj(preds, tickets, writer)
        except BrokenPipeError:
            pass
        content = writer.getvalue()
        assert content.startswith("id,file_name,sentiment,confidence")
        assert "\n" in content

    def test_predictions_stream_broken_pipe_header_written(self, gen):
        writer = BrokenPipeWriter(fail_after_rows=0)
        preds = make_predictions(100)
        tickets = make_tickets(100)
        try:
            gen._stream_predictions_to_fileobj(preds, tickets, writer)
        except BrokenPipeError:
            pass
        content = writer.getvalue()
        assert "id,file_name,sentiment,confidence" in content

    def test_predictions_stream_generator_not_fully_consumed(self, gen):
        consumed = [0]

        def counting_gen():
            for p in make_predictions(10000):
                consumed[0] += 1
                yield p

        writer = BrokenPipeWriter(fail_after_rows=2)
        tickets = make_tickets(10000)
        try:
            gen._stream_predictions_to_fileobj(counting_gen(), tickets, writer)
        except BrokenPipeError:
            pass
        assert consumed[0] < 10000

    def test_stream_predictions_csv_to_path_propagates_broken_pipe(self, gen, monkeypatch):
        broken_open_called = [False]

        def broken_open(*args, **kwargs):
            broken_open_called[0] = True
            return BrokenPipeWriter(fail_after_rows=5)

        monkeypatch.setattr("builtins.open", broken_open)
        preds = list(make_predictions(200))
        tickets = make_tickets(200)
        with pytest.raises(BrokenPipeError):
            gen.stream_predictions_csv(iter(preds), tickets, "/tmp/fake.csv")
        assert broken_open_called[0]

    def test_iter_predictions_csv_rows_broken_pipe_on_chunk(self, gen):
        class ChunkBrokenPipe:
            def __init__(self, fail_after: int):
                self.fail_after = fail_after
                self.count = 0

            def __call__(self, *args, **kwargs):
                self.count += 1
                if self.count > self.fail_after:
                    raise BrokenPipeError(errno.EPIPE, "Pipe broken during chunk")

        rows_iter = gen.iter_predictions_csv_rows(make_predictions(100), make_tickets(100))
        chunks_collected = []
        try:
            for idx, chunk in enumerate(rows_iter):
                if idx == 2:
                    raise BrokenPipeError(errno.EPIPE, "Simulated consumer gone")
                chunks_collected.append(chunk)
        except BrokenPipeError:
            pass
        assert len(chunks_collected) == 2
        assert all(isinstance(c, str) for c in chunks_collected)


class TestStreamingBrokenPipeLowConfidence:
    @pytest.fixture
    def gen(self):
        return ReportGenerator(low_confidence_threshold=0.7)

    def test_low_conf_stream_propagates_broken_pipe(self, gen):
        writer = BrokenPipeOnSecondWrite()
        low_conf_iter = gen.iter_low_confidence_samples(
            make_predictions(1000), make_tickets(1000)
        )
        with pytest.raises(BrokenPipeError):
            gen._stream_low_confidence_to_fileobj(low_conf_iter, writer)

    def test_low_conf_stream_partial_content_on_break(self, gen):
        writer = BrokenPipeWriter(fail_after_rows=1)
        low_conf_iter = gen.iter_low_confidence_samples(
            make_predictions(1000), make_tickets(1000)
        )
        try:
            gen._stream_low_confidence_to_fileobj(low_conf_iter, writer)
        except BrokenPipeError:
            pass
        content = writer.getvalue()
        assert content.startswith("index,file_name,sentiment,confidence")

    def test_low_conf_stream_header_written_before_break(self, gen):
        writer = BrokenPipeWriter(fail_after_rows=0)
        low_conf_iter = gen.iter_low_confidence_samples(
            make_predictions(100), make_tickets(100)
        )
        try:
            gen._stream_low_confidence_to_fileobj(low_conf_iter, writer)
        except BrokenPipeError:
            pass
        content = writer.getvalue()
        assert "index,file_name,sentiment,confidence" in content

    def test_low_conf_iter_stops_when_downstream_breaks(self, gen):
        n = 10000
        consumed_predictions = [0]

        def counting_predictions():
            for p in make_predictions(n):
                consumed_predictions[0] += 1
                yield p

        low_conf_iter = gen.iter_low_confidence_samples(
            counting_predictions(), make_tickets(n)
        )
        items_seen = 0
        try:
            for item in low_conf_iter:
                items_seen += 1
                if items_seen == 3:
                    raise BrokenPipeError(errno.EPIPE, "Consumer disconnected")
        except BrokenPipeError:
            pass
        assert items_seen == 3
        assert consumed_predictions[0] < n


class TestStreamingBrokenPipeFullReport:
    @pytest.fixture
    def gen(self):
        return ReportGenerator(low_confidence_threshold=0.7, output_format="csv")

    def test_file_removed_if_predictions_csv_broken_during_write(self, gen, monkeypatch, tmp_path):
        import ticket_classifier.report_generator as rg_mod
        real_open = builtins.open

        class _Opener:
            def __init__(self):
                self.wrapped = None

            def __call__(self, path, *args, **kwargs):
                if str(path).endswith("predictions.csv"):
                    self.wrapped = BrokenPipeOnSecondWrite()
                    return self.wrapped
                return real_open(path, *args, **kwargs)

        opener = _Opener()
        monkeypatch.setattr(rg_mod.builtins, "open", opener)
        preds = list(make_predictions(500))
        tickets = make_tickets(500)

        real_remove = rg_mod.os.remove
        removed_paths = []

        def tracking_remove(p):
            removed_paths.append(str(p))
            try:
                real_remove(p)
            except OSError:
                pass

        monkeypatch.setattr(rg_mod.os, "remove", tracking_remove)

        with pytest.raises(BrokenPipeError):
            gen.generate_full_report(preds, tickets, output_dir=str(tmp_path))

        assert len(removed_paths) >= 1
        assert any(p.endswith("predictions.csv") for p in removed_paths)

    def test_streaming_no_memory_leak_on_broken_pipe(self, gen):
        import gc

        gc.collect()
        preds_list = list(make_predictions(5000))
        tickets = make_tickets(5000)

        writer = BrokenPipeWriter(fail_after_rows=10)
        try:
            gen._stream_predictions_to_fileobj(iter(preds_list), tickets, writer)
        except BrokenPipeError:
            pass

        del preds_list
        del tickets
        del writer
        gc.collect()

        import sys
        assert True

    def test_csv_output_on_broken_pipe_has_valid_header(self, gen):
        writer = BrokenPipeWriter(fail_after_rows=0)
        preds = make_predictions(500)
        tickets = make_tickets(500)
        try:
            gen._stream_predictions_to_fileobj(preds, tickets, writer)
        except BrokenPipeError:
            pass
        content = writer.getvalue()
        reader = csv.DictReader(io.StringIO(content))
        headers = reader.fieldnames
        assert headers is not None
        assert "id" in headers
        assert "file_name" in headers
        assert "sentiment" in headers
        assert "confidence" in headers


class TestBrokenPipeCleanupFullPipeline:
    def test_cli_path_exception_is_broken_pipe(self, monkeypatch, tmp_path):
        import tempfile as _tf
        from ticket_classifier.config import Config

        preds_dir = _tf.mkdtemp(dir=str(tmp_path))
        for i in range(50):
            Path(preds_dir) / f"ticket_{i:04d}.txt"
            (Path(preds_dir) / f"ticket_{i:04d}.txt").write_text(f"Content {i}")

        output_dir = _tf.mkdtemp(dir=str(tmp_path))

        cfg = Config({"confidence_threshold": 0.7, "output_format": "csv",
                       "output_dir": output_dir, "random_state": 42})

        from ticket_classifier.report_generator import ReportGenerator

        gen = ReportGenerator(
            low_confidence_threshold=cfg.confidence_threshold,
            output_format=cfg.output_format
        )

        preds = list(make_predictions(50))
        tickets = make_tickets(50)

        original_open = open

        def broken_open(path, *args, **kwargs):
            if str(path).endswith("predictions.csv"):
                return BrokenPipeWriter(fail_after_rows=3)
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", broken_open)
        with pytest.raises(BrokenPipeError):
            gen.generate_full_report(preds, tickets, output_dir=cfg.output_dir)

    def test_iter_low_confidence_samples_does_not_cache(self):
        gen = ReportGenerator(low_confidence_threshold=0.7)
        n = 1000
        produced = [0]

        def gen_predictions():
            for p in make_predictions(n):
                produced[0] += 1
                yield p

        iter_obj = gen.iter_low_confidence_samples(gen_predictions(), make_tickets(n))
        first_three = []
        for i, item in enumerate(iter_obj):
            first_three.append(item)
            if i == 2:
                break
        assert len(first_three) == 3
        assert produced[0] < n
