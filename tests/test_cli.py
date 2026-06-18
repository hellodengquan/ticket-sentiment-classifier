import pytest
import json
import tempfile
import os
from pathlib import Path
from ticket_classifier.cli import build_parser, load_config, run
from ticket_classifier.config import Config


class TestBuildParser:
    def test_parser_creation(self):
        parser = build_parser()
        assert parser is not None

    def test_parser_required_input(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_parser_input_path(self):
        parser = build_parser()
        args = parser.parse_args(["/some/path"])
        assert args.input_path == "/some/path"

    def test_parser_confidence_threshold(self):
        parser = build_parser()
        args = parser.parse_args(["/path", "-t", "0.8"])
        assert args.confidence_threshold == 0.8

    def test_parser_confidence_threshold_long(self):
        parser = build_parser()
        args = parser.parse_args(["/path", "--confidence-threshold", "0.6"])
        assert args.confidence_threshold == 0.6

    def test_parser_output_format(self):
        for fmt in ("json", "csv", "both"):
            parser = build_parser()
            args = parser.parse_args(["/path", "-f", fmt])
            assert args.output_format == fmt

    def test_parser_output_format_invalid(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["/path", "-f", "xml"])

    def test_parser_output_dir(self):
        parser = build_parser()
        args = parser.parse_args(["/path", "-o", "/tmp/out"])
        assert args.output_dir == "/tmp/out"

    def test_parser_config_file(self):
        parser = build_parser()
        args = parser.parse_args(["/path", "-c", "config.json"])
        assert args.config == "config.json"

    def test_parser_random_state(self):
        parser = build_parser()
        args = parser.parse_args(["/path", "--random-state", "123"])
        assert args.random_state == 123

    def test_parser_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["/path"])
        assert args.confidence_threshold is None
        assert args.output_format is None
        assert args.output_dir is None
        assert args.config is None
        assert args.random_state is None


class TestLoadConfig:
    def test_load_config_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["/path"])
        config = load_config(args)
        assert config.confidence_threshold == 0.7
        assert config.output_format == "both"

    def test_load_config_cli_threshold(self):
        parser = build_parser()
        args = parser.parse_args(["/path", "-t", "0.85"])
        config = load_config(args)
        assert config.confidence_threshold == 0.85

    def test_load_config_cli_format(self):
        parser = build_parser()
        args = parser.parse_args(["/path", "-f", "csv"])
        config = load_config(args)
        assert config.output_format == "csv"

    def test_load_config_cli_overrides_config_file(self):
        config_data = {
            "confidence_threshold": 0.5,
            "output_format": "json"
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            parser = build_parser()
            args = parser.parse_args(["/path", "-c", temp_path, "-t", "0.9"])
            config = load_config(args)
            assert config.confidence_threshold == 0.9
            assert config.output_format == "json"
        finally:
            os.unlink(temp_path)

    def test_load_config_from_file(self):
        config_data = {
            "confidence_threshold": 0.55,
            "output_format": "csv",
            "output_dir": "/custom/out"
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            parser = build_parser()
            args = parser.parse_args(["/path", "-c", temp_path])
            config = load_config(args)
            assert config.confidence_threshold == 0.55
            assert config.output_format == "csv"
            assert config.output_dir == "/custom/out"
        finally:
            os.unlink(temp_path)


class TestCLIRun:
    @pytest.fixture
    def sample_tickets_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tickets = [
                ("ticket_001.txt", "This product is absolutely wonderful! Best purchase ever."),
                ("ticket_002.txt", "Terrible experience, the product broke immediately."),
                ("ticket_003.txt", "The product is okay, works as advertised."),
            ]
            for filename, content in tickets:
                (Path(temp_dir) / filename).write_text(content, encoding="utf-8")
            yield temp_dir

    def test_run_with_directory(self, sample_tickets_dir):
        exit_code = run([sample_tickets_dir])
        assert exit_code == 0

    def test_run_with_single_file(self, sample_tickets_dir):
        file_path = str(Path(sample_tickets_dir) / "ticket_001.txt")
        exit_code = run([file_path])
        assert exit_code == 0

    def test_run_with_threshold(self, sample_tickets_dir):
        exit_code = run([sample_tickets_dir, "-t", "0.5"])
        assert exit_code == 0

    def test_run_with_output_format_json(self, sample_tickets_dir):
        with tempfile.TemporaryDirectory() as output_dir:
            exit_code = run([sample_tickets_dir, "-f", "json", "-o", output_dir])
            assert exit_code == 0

    def test_run_with_output_format_csv(self, sample_tickets_dir):
        with tempfile.TemporaryDirectory() as output_dir:
            exit_code = run([sample_tickets_dir, "-f", "csv", "-o", output_dir])
            assert exit_code == 0

    def test_run_with_output_format_both(self, sample_tickets_dir):
        with tempfile.TemporaryDirectory() as output_dir:
            exit_code = run([sample_tickets_dir, "-f", "both", "-o", output_dir])
            assert exit_code == 0
            assert (Path(output_dir) / "batch_summary.json").exists()
            assert (Path(output_dir) / "low_confidence_report.json").exists()
            assert (Path(output_dir) / "predictions.csv").exists()
            assert (Path(output_dir) / "low_confidence.csv").exists()

    def test_run_nonexistent_path(self):
        exit_code = run(["/nonexistent/path"])
        assert exit_code == 1

    def test_run_with_config_file(self, sample_tickets_dir):
        config_data = {"confidence_threshold": 0.6, "output_format": "json"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            with tempfile.TemporaryDirectory() as output_dir:
                exit_code = run([sample_tickets_dir, "-c", config_path, "-o", output_dir])
                assert exit_code == 0
        finally:
            os.unlink(config_path)

    def test_run_csv_only_output(self, sample_tickets_dir):
        with tempfile.TemporaryDirectory() as output_dir:
            exit_code = run([sample_tickets_dir, "-f", "csv", "-o", output_dir])
            assert exit_code == 0
            assert (Path(output_dir) / "predictions.csv").exists()
            assert (Path(output_dir) / "low_confidence.csv").exists()
            assert not (Path(output_dir) / "batch_summary.json").exists()

    def test_run_json_only_output(self, sample_tickets_dir):
        with tempfile.TemporaryDirectory() as output_dir:
            exit_code = run([sample_tickets_dir, "-f", "json", "-o", output_dir])
            assert exit_code == 0
            assert (Path(output_dir) / "batch_summary.json").exists()
            assert (Path(output_dir) / "low_confidence_report.json").exists()
            assert not (Path(output_dir) / "predictions.csv").exists()
