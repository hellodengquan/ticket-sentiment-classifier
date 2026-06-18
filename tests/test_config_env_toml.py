import pytest
import os
import json
import tempfile
from pathlib import Path
from ticket_classifier.config import Config, ENV_PREFIX, ENV_KEY_MAP, DEFAULT_CONFIG


ENV_KEY_CONFIDENCE = f"{ENV_PREFIX}CONFIDENCE_THRESHOLD"
ENV_KEY_FORMAT = f"{ENV_PREFIX}OUTPUT_FORMAT"
ENV_KEY_DIR = f"{ENV_PREFIX}OUTPUT_DIR"
ENV_KEY_RANDOM = f"{ENV_PREFIX}RANDOM_STATE"

ENV_KEY_DIR_WITH_PREFIX = "env"  # placeholder, we'll use ENV_KEY_DIR directly


class TestEnvVarFallback:
    def test_from_env_empty(self, monkeypatch):
        for key in ENV_KEY_MAP:
            monkeypatch.delenv(key, raising=False)
        config = Config.from_env()
        assert config.confidence_threshold == DEFAULT_CONFIG["confidence_threshold"]
        assert config.output_format == DEFAULT_CONFIG["output_format"]

    def test_from_env_confidence_threshold(self, monkeypatch):
        for key in ENV_KEY_MAP:
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv(ENV_KEY_CONFIDENCE, "0.85")
        config = Config.from_env()
        assert config.confidence_threshold == 0.85

    def test_from_env_output_format(self, monkeypatch):
        for key in ENV_KEY_MAP:
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv(ENV_KEY_FORMAT, "csv")
        config = Config.from_env()
        assert config.output_format == "csv"

    def test_from_env_output_dir(self, monkeypatch):
        for key in ENV_KEY_MAP:
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv(ENV_KEY_DIR, "/env/reports")
        config = Config.from_env()
        assert config.output_dir == "/env/reports"

    def test_from_env_random_state(self, monkeypatch):
        for key in ENV_KEY_MAP:
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv(ENV_KEY_RANDOM, "77")
        config = Config.from_env()
        assert config.random_state == 77

    def test_from_env_all_vars(self, monkeypatch):
        monkeypatch.setenv(ENV_KEY_CONFIDENCE, "0.6")
        monkeypatch.setenv(ENV_KEY_FORMAT, "json")
        monkeypatch.setenv(ENV_KEY_DIR, "/custom/out")
        monkeypatch.setenv(ENV_KEY_RANDOM, "101")
        config = Config.from_env()
        assert config.confidence_threshold == 0.6
        assert config.output_format == "json"
        assert config.output_dir == "/custom/out"
        assert config.random_state == 101

    def test_from_env_invalid_confidence(self, monkeypatch):
        for key in ENV_KEY_MAP:
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv(ENV_KEY_CONFIDENCE, "not_a_number")
        with pytest.raises(ValueError, match="must be a valid float"):
            Config.from_env()

    def test_from_env_invalid_random_state(self, monkeypatch):
        for key in ENV_KEY_MAP:
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv(ENV_KEY_RANDOM, "abc")
        with pytest.raises(ValueError, match="must be a valid integer"):
            Config.from_env()


class TestTomlConfig:
    def test_from_toml_file(self):
        toml_content = """
[ticket_classifier]
confidence_threshold = 0.65
output_format = "csv"
output_dir = "/toml/out"
random_state = 55
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            temp_path = f.name

        try:
            config = Config.from_file(temp_path)
            assert config.confidence_threshold == 0.65
            assert config.output_format == "csv"
            assert config.output_dir == "/toml/out"
            assert config.random_state == 55
        finally:
            os.unlink(temp_path)

    def test_from_toml_flat(self):
        toml_content = """
confidence_threshold = 0.75
output_format = "both"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            temp_path = f.name

        try:
            config = Config.from_file(temp_path)
            assert config.confidence_threshold == 0.75
            assert config.output_format == "both"
        finally:
            os.unlink(temp_path)

    def test_from_toml_partial(self):
        toml_content = """
[ticket_classifier]
confidence_threshold = 0.9
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            temp_path = f.name

        try:
            config = Config.from_file(temp_path)
            assert config.confidence_threshold == 0.9
            assert config.output_format == DEFAULT_CONFIG["output_format"]
            assert config.random_state == DEFAULT_CONFIG["random_state"]
        finally:
            os.unlink(temp_path)

    def test_from_toml_not_found(self):
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            Config.from_file("/nonexistent/config.toml")

    def test_to_toml_file(self):
        config = Config({"confidence_threshold": 0.88, "output_format": "json"})
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "config.toml")
            config.to_file(file_path)
            assert os.path.exists(file_path)

            loaded = Config.from_file(file_path)
            assert loaded.confidence_threshold == 0.88
            assert loaded.output_format == "json"

    def test_unsupported_config_format(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("key: value")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported config file format"):
                Config.from_file(temp_path)
        finally:
            os.unlink(temp_path)

    def test_to_file_unsupported_format(self):
        config = Config()
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "config.yaml")
            with pytest.raises(ValueError, match="Unsupported config file format"):
                config.to_file(file_path)


class TestResolvePriority:
    def test_default_only(self, monkeypatch):
        for key in ENV_KEY_MAP:
            monkeypatch.delenv(key, raising=False)
        config = Config.resolve(cli_args=None, config_file_path=None)
        assert config.confidence_threshold == DEFAULT_CONFIG["confidence_threshold"]
        assert config.output_format == DEFAULT_CONFIG["output_format"]
        assert config.output_dir == DEFAULT_CONFIG["output_dir"]

    def test_env_overrides_default(self, monkeypatch):
        for key in ENV_KEY_MAP:
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv(ENV_KEY_CONFIDENCE, "0.5")
        monkeypatch.setenv(ENV_KEY_FORMAT, "csv")
        config = Config.resolve(cli_args=None, config_file_path=None)
        assert config.confidence_threshold == 0.5
        assert config.output_format == "csv"

    def test_file_overrides_env(self, monkeypatch):
        for key in ENV_KEY_MAP:
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv(ENV_KEY_CONFIDENCE, "0.5")

        config_data = {"confidence_threshold": 0.8}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            config = Config.resolve(cli_args=None, config_file_path=temp_path)
            assert config.confidence_threshold == 0.8
        finally:
            os.unlink(temp_path)

    def test_cli_overrides_file_and_env(self, monkeypatch):
        for key in ENV_KEY_MAP:
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv(ENV_KEY_CONFIDENCE, "0.3")

        config_data = {"confidence_threshold": 0.6}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            class MockArgs:
                confidence_threshold = 0.95
                output_format = None
                output_dir = None
                random_state = None

            config = Config.resolve(cli_args=MockArgs(), config_file_path=temp_path)
            assert config.confidence_threshold == 0.95
        finally:
            os.unlink(temp_path)

    def test_cli_all_args(self, monkeypatch):
        for key in ENV_KEY_MAP:
            monkeypatch.delenv(key, raising=False)

        class MockArgs:
            confidence_threshold = 0.4
            output_format = "json"
            output_dir = "/cli/out"
            random_state = 4242

        config = Config.resolve(cli_args=MockArgs(), config_file_path=None)
        assert config.confidence_threshold == 0.4
        assert config.output_format == "json"
        assert config.output_dir == "/cli/out"
        assert config.random_state == 4242

    def test_toml_file_overrides_env(self, monkeypatch):
        for key in ENV_KEY_MAP:
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv(ENV_KEY_FORMAT, "csv")

        toml_content = """
[ticket_classifier]
output_format = "json"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            temp_path = f.name

        try:
            config = Config.resolve(cli_args=None, config_file_path=temp_path)
            assert config.output_format == "json"
        finally:
            os.unlink(temp_path)

    def test_cli_overrides_toml(self, monkeypatch):
        for key in ENV_KEY_MAP:
            monkeypatch.delenv(key, raising=False)

        toml_content = """
[ticket_classifier]
confidence_threshold = 0.2
random_state = 100
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            temp_path = f.name

        try:
            class MockArgs:
                confidence_threshold = 0.9
                output_format = None
                output_dir = None
                random_state = 500

            config = Config.resolve(cli_args=MockArgs(), config_file_path=temp_path)
            assert config.confidence_threshold == 0.9
            assert config.random_state == 500
        finally:
            os.unlink(temp_path)

    def test_full_priority_chain(self, monkeypatch):
        for key in ENV_KEY_MAP:
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv(ENV_KEY_CONFIDENCE, "0.1")
        monkeypatch.setenv(ENV_KEY_DIR, "/env/dir")

        toml_content = """
[ticket_classifier]
confidence_threshold = 0.2
output_format = "csv"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            temp_path = f.name

        try:
            class MockArgs:
                confidence_threshold = 0.3
                output_format = None
                output_dir = None
                random_state = None

            config = Config.resolve(cli_args=MockArgs(), config_file_path=temp_path)
            assert config.confidence_threshold == 0.3
            assert config.output_format == "csv"
            assert config.output_dir == "/env/dir"
        finally:
            os.unlink(temp_path)
