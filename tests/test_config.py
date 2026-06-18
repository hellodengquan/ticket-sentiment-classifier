import pytest
import tempfile
import os
import json
from ticket_classifier.config import Config, DEFAULT_CONFIG


class TestConfig:
    def test_default_values(self):
        config = Config()
        assert config.confidence_threshold == DEFAULT_CONFIG["confidence_threshold"]
        assert config.output_format == DEFAULT_CONFIG["output_format"]
        assert config.output_dir == DEFAULT_CONFIG["output_dir"]
        assert config.random_state == DEFAULT_CONFIG["random_state"]

    def test_custom_values_via_constructor(self):
        config = Config({
            "confidence_threshold": 0.5,
            "output_format": "csv",
            "output_dir": "/tmp/reports",
            "random_state": 123
        })
        assert config.confidence_threshold == 0.5
        assert config.output_format == "csv"
        assert config.output_dir == "/tmp/reports"
        assert config.random_state == 123

    def test_confidence_threshold_setter(self):
        config = Config()
        config.confidence_threshold = 0.8
        assert config.confidence_threshold == 0.8

    def test_confidence_threshold_invalid_zero(self):
        config = Config()
        with pytest.raises(ValueError, match="confidence_threshold must be in"):
            config.confidence_threshold = 0.0

    def test_confidence_threshold_invalid_negative(self):
        config = Config()
        with pytest.raises(ValueError, match="confidence_threshold must be in"):
            config.confidence_threshold = -0.5

    def test_confidence_threshold_invalid_above_one(self):
        config = Config()
        with pytest.raises(ValueError, match="confidence_threshold must be in"):
            config.confidence_threshold = 1.5

    def test_confidence_threshold_valid_one(self):
        config = Config()
        config.confidence_threshold = 1.0
        assert config.confidence_threshold == 1.0

    def test_output_format_valid_values(self):
        for fmt in ("json", "csv", "both"):
            config = Config()
            config.output_format = fmt
            assert config.output_format == fmt

    def test_output_format_invalid(self):
        config = Config()
        with pytest.raises(ValueError, match="output_format must be one of"):
            config.output_format = "xml"

    def test_output_dir_setter(self):
        config = Config()
        config.output_dir = "/custom/dir"
        assert config.output_dir == "/custom/dir"

    def test_random_state_setter(self):
        config = Config()
        config.random_state = 99
        assert config.random_state == 99

    def test_to_dict(self):
        config = Config({"confidence_threshold": 0.6})
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["confidence_threshold"] == 0.6
        assert "output_format" in d

    def test_to_dict_returns_copy(self):
        config = Config()
        d = config.to_dict()
        d["confidence_threshold"] = 0.1
        assert config.confidence_threshold != 0.1

    def test_from_file(self):
        config_data = {
            "confidence_threshold": 0.55,
            "output_format": "json",
            "output_dir": "/tmp/test_output",
            "random_state": 7
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            config = Config.from_file(temp_path)
            assert config.confidence_threshold == 0.55
            assert config.output_format == "json"
            assert config.output_dir == "/tmp/test_output"
            assert config.random_state == 7
        finally:
            os.unlink(temp_path)

    def test_from_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            Config.from_file("/nonexistent/config.json")

    def test_from_file_partial_config(self):
        config_data = {"confidence_threshold": 0.9}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            config = Config.from_file(temp_path)
            assert config.confidence_threshold == 0.9
            assert config.output_format == DEFAULT_CONFIG["output_format"]
        finally:
            os.unlink(temp_path)

    def test_to_file(self):
        config = Config({"confidence_threshold": 0.65, "output_format": "csv"})
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "subdir", "config.json")
            config.to_file(file_path)

            assert os.path.exists(file_path)
            with open(file_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            assert loaded["confidence_threshold"] == 0.65
            assert loaded["output_format"] == "csv"

    def test_roundtrip_file(self):
        original = Config({"confidence_threshold": 0.75, "output_format": "both", "random_state": 42})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            original.to_file(temp_path)
            loaded = Config.from_file(temp_path)
            assert loaded.to_dict() == original.to_dict()
        finally:
            os.unlink(temp_path)

    def test_from_cli_args(self):
        class MockArgs:
            confidence_threshold = 0.85
            output_format = "csv"
            output_dir = "/cli/output"
            random_state = 100

        config = Config.from_cli_args(MockArgs())
        assert config.confidence_threshold == 0.85
        assert config.output_format == "csv"
        assert config.output_dir == "/cli/output"
        assert config.random_state == 100

    def test_from_cli_args_none_values(self):
        class MockArgs:
            confidence_threshold = None
            output_format = None
            output_dir = None
            random_state = None

        config = Config.from_cli_args(MockArgs())
        assert config.confidence_threshold == DEFAULT_CONFIG["confidence_threshold"]
        assert config.output_format == DEFAULT_CONFIG["output_format"]

    def test_from_cli_args_partial(self):
        class MockArgs:
            confidence_threshold = 0.6
            output_format = None
            output_dir = None
            random_state = None

        config = Config.from_cli_args(MockArgs())
        assert config.confidence_threshold == 0.6
        assert config.output_format == DEFAULT_CONFIG["output_format"]
