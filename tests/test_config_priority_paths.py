import pytest
import json
import tempfile
import os
from ticket_classifier.config import Config, ENV_PREFIX, DEFAULT_CONFIG


ENV_KEY_CONFIDENCE = f"{ENV_PREFIX}CONFIDENCE_THRESHOLD"
ENV_KEY_FORMAT = f"{ENV_PREFIX}OUTPUT_FORMAT"
ENV_KEY_DIR = f"{ENV_PREFIX}OUTPUT_DIR"
ENV_KEY_RANDOM = f"{ENV_PREFIX}RANDOM_STATE"


def _make_mock_args(**kwargs):
    defaults = {
        "confidence_threshold": None,
        "output_format": None,
        "output_dir": None,
        "random_state": None,
    }
    defaults.update(kwargs)

    class _Args:
        pass

    obj = _Args()
    for k, v in defaults.items():
        setattr(obj, k, v)
    return obj


def _write_json_config(tmp_path, **kwargs):
    data = dict(DEFAULT_CONFIG)
    data.update(kwargs)
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", dir=str(tmp_path), delete=False)
    json.dump(data, f)
    f.close()
    return f.name


def _write_toml_config(tmp_path, **kwargs):
    try:
        import tomli_w
    except ImportError:
        pytest.skip("tomli-w not available")
    data = dict(DEFAULT_CONFIG)
    data.update(kwargs)
    f = tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", dir=str(tmp_path), delete=False)
    tomli_w.dump({"ticket_classifier": data}, f)
    f.close()
    return f.name


class TestPriorityPerKey:
    def test_confidence_cli_over_file_over_env(self, monkeypatch, tmp_path):
        for k in [ENV_KEY_CONFIDENCE, ENV_KEY_FORMAT, ENV_KEY_DIR, ENV_KEY_RANDOM]:
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv(ENV_KEY_CONFIDENCE, "0.1")
        cfg_path = _write_json_config(tmp_path, confidence_threshold=0.2)
        args = _make_mock_args(confidence_threshold=0.3)

        cfg = Config.resolve(cli_args=args, config_file_path=cfg_path)
        assert cfg.confidence_threshold == 0.3

    def test_confidence_file_over_env(self, monkeypatch, tmp_path):
        for k in [ENV_KEY_CONFIDENCE, ENV_KEY_FORMAT, ENV_KEY_DIR, ENV_KEY_RANDOM]:
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv(ENV_KEY_CONFIDENCE, "0.1")
        cfg_path = _write_json_config(tmp_path, confidence_threshold=0.2)

        cfg = Config.resolve(cli_args=None, config_file_path=cfg_path)
        assert cfg.confidence_threshold == 0.2

    def test_confidence_env_only(self, monkeypatch):
        for k in [ENV_KEY_CONFIDENCE, ENV_KEY_FORMAT, ENV_KEY_DIR, ENV_KEY_RANDOM]:
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv(ENV_KEY_CONFIDENCE, "0.15")

        cfg = Config.resolve(cli_args=None, config_file_path=None)
        assert cfg.confidence_threshold == 0.15

    def test_confidence_cli_only(self):
        args = _make_mock_args(confidence_threshold=0.42)
        cfg = Config.resolve(cli_args=args, config_file_path=None)
        assert cfg.confidence_threshold == 0.42

    def test_confidence_default(self):
        cfg = Config.resolve(cli_args=None, config_file_path=None)
        assert cfg.confidence_threshold == DEFAULT_CONFIG["confidence_threshold"]

    def test_format_cli_over_toml_over_env(self, monkeypatch, tmp_path):
        for k in [ENV_KEY_CONFIDENCE, ENV_KEY_FORMAT, ENV_KEY_DIR, ENV_KEY_RANDOM]:
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv(ENV_KEY_FORMAT, "json")
        cfg_path = _write_toml_config(tmp_path, output_format="csv")
        args = _make_mock_args(output_format="both")

        cfg = Config.resolve(cli_args=args, config_file_path=cfg_path)
        assert cfg.output_format == "both"

    def test_format_toml_over_env(self, monkeypatch, tmp_path):
        for k in [ENV_KEY_CONFIDENCE, ENV_KEY_FORMAT, ENV_KEY_DIR, ENV_KEY_RANDOM]:
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv(ENV_KEY_FORMAT, "json")
        cfg_path = _write_toml_config(tmp_path, output_format="csv")

        cfg = Config.resolve(cli_args=None, config_file_path=cfg_path)
        assert cfg.output_format == "csv"

    def test_output_dir_cli_over_json_over_env(self, monkeypatch, tmp_path):
        for k in [ENV_KEY_CONFIDENCE, ENV_KEY_FORMAT, ENV_KEY_DIR, ENV_KEY_RANDOM]:
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv(ENV_KEY_DIR, "/env/dir")
        cfg_path = _write_json_config(tmp_path, output_dir="/json/dir")
        args = _make_mock_args(output_dir="/cli/dir")

        cfg = Config.resolve(cli_args=args, config_file_path=cfg_path)
        assert cfg.output_dir == "/cli/dir"

    def test_output_dir_json_over_env(self, monkeypatch, tmp_path):
        for k in [ENV_KEY_CONFIDENCE, ENV_KEY_FORMAT, ENV_KEY_DIR, ENV_KEY_RANDOM]:
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv(ENV_KEY_DIR, "/env/dir")
        cfg_path = _write_json_config(tmp_path, output_dir="/json/dir")

        cfg = Config.resolve(cli_args=None, config_file_path=cfg_path)
        assert cfg.output_dir == "/json/dir"

    def test_output_dir_env_fallback_when_file_is_default(self, monkeypatch, tmp_path):
        for k in [ENV_KEY_CONFIDENCE, ENV_KEY_FORMAT, ENV_KEY_DIR, ENV_KEY_RANDOM]:
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv(ENV_KEY_DIR, "/env/dir")
        cfg_path = _write_json_config(tmp_path, confidence_threshold=0.5)

        cfg = Config.resolve(cli_args=None, config_file_path=cfg_path)
        assert cfg.output_dir == "/env/dir"

    def test_random_state_cli_over_toml_over_env(self, monkeypatch, tmp_path):
        for k in [ENV_KEY_CONFIDENCE, ENV_KEY_FORMAT, ENV_KEY_DIR, ENV_KEY_RANDOM]:
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv(ENV_KEY_RANDOM, "10")
        cfg_path = _write_toml_config(tmp_path, random_state=20)
        args = _make_mock_args(random_state=30)

        cfg = Config.resolve(cli_args=args, config_file_path=cfg_path)
        assert cfg.random_state == 30


class TestPriorityAllKeysSimultaneously:
    def test_three_sources_each_key_different(self, monkeypatch, tmp_path):
        for k in [ENV_KEY_CONFIDENCE, ENV_KEY_FORMAT, ENV_KEY_DIR, ENV_KEY_RANDOM]:
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv(ENV_KEY_CONFIDENCE, "0.1")
        monkeypatch.setenv(ENV_KEY_FORMAT, "json")
        monkeypatch.setenv(ENV_KEY_DIR, "/env/dir")
        monkeypatch.setenv(ENV_KEY_RANDOM, "1")

        cfg_path = _write_json_config(
            tmp_path,
            confidence_threshold=0.2,
            output_format="csv",
            output_dir="/json/dir",
            random_state=2,
        )
        args = _make_mock_args(
            confidence_threshold=0.3,
            output_format="both",
            output_dir="/cli/dir",
            random_state=3,
        )

        cfg = Config.resolve(cli_args=args, config_file_path=cfg_path)
        assert cfg.confidence_threshold == 0.3
        assert cfg.output_format == "both"
        assert cfg.output_dir == "/cli/dir"
        assert cfg.random_state == 3

    def test_mixed_override_per_key(self, monkeypatch, tmp_path):
        for k in [ENV_KEY_CONFIDENCE, ENV_KEY_FORMAT, ENV_KEY_DIR, ENV_KEY_RANDOM]:
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv(ENV_KEY_CONFIDENCE, "0.15")
        monkeypatch.setenv(ENV_KEY_FORMAT, "csv")
        monkeypatch.setenv(ENV_KEY_DIR, "/env/dir")
        monkeypatch.setenv(ENV_KEY_RANDOM, "99")

        cfg_path = _write_toml_config(
            tmp_path,
            confidence_threshold=0.25,
            output_dir="/toml/dir",
        )
        args = _make_mock_args(
            output_format="both",
        )

        cfg = Config.resolve(cli_args=args, config_file_path=cfg_path)
        assert cfg.confidence_threshold == 0.25
        assert cfg.output_format == "both"
        assert cfg.output_dir == "/toml/dir"
        assert cfg.random_state == 99

    def test_file_has_some_keys_env_fills_rest(self, monkeypatch, tmp_path):
        for k in [ENV_KEY_CONFIDENCE, ENV_KEY_FORMAT, ENV_KEY_DIR, ENV_KEY_RANDOM]:
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv(ENV_KEY_DIR, "/env/dir")
        monkeypatch.setenv(ENV_KEY_RANDOM, "4242")

        cfg_path = _write_json_config(tmp_path, confidence_threshold=0.5, output_format="json")

        cfg = Config.resolve(cli_args=None, config_file_path=cfg_path)
        assert cfg.confidence_threshold == 0.5
        assert cfg.output_format == "json"
        assert cfg.output_dir == "/env/dir"
        assert cfg.random_state == 4242

    def test_cli_partial_overrides(self, monkeypatch, tmp_path):
        for k in [ENV_KEY_CONFIDENCE, ENV_KEY_FORMAT, ENV_KEY_DIR, ENV_KEY_RANDOM]:
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv(ENV_KEY_CONFIDENCE, "0.1")

        cfg_path = _write_toml_config(tmp_path, output_format="csv")
        args = _make_mock_args(confidence_threshold=0.9)

        cfg = Config.resolve(cli_args=args, config_file_path=cfg_path)
        assert cfg.confidence_threshold == 0.9
        assert cfg.output_format == "csv"


class TestPriorityEdgeCases:
    def test_cli_args_none_equivalent_to_no_cli(self, monkeypatch):
        for k in [ENV_KEY_CONFIDENCE, ENV_KEY_FORMAT, ENV_KEY_DIR, ENV_KEY_RANDOM]:
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv(ENV_KEY_CONFIDENCE, "0.55")
        args = _make_mock_args()
        cfg_with = Config.resolve(cli_args=args, config_file_path=None)
        cfg_without = Config.resolve(cli_args=None, config_file_path=None)
        assert cfg_with.to_dict() == cfg_without.to_dict()
        assert cfg_with.confidence_threshold == 0.55

    def test_file_does_not_override_env_when_file_sets_default(self, monkeypatch, tmp_path):
        for k in [ENV_KEY_CONFIDENCE, ENV_KEY_FORMAT, ENV_KEY_DIR, ENV_KEY_RANDOM]:
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv(ENV_KEY_DIR, "/custom/env")
        cfg_path = _write_json_config(
            tmp_path,
            output_dir=DEFAULT_CONFIG["output_dir"],
            confidence_threshold=0.5,
        )

        cfg = Config.resolve(cli_args=None, config_file_path=cfg_path)
        assert cfg.output_dir == "/custom/env"
        assert cfg.confidence_threshold == 0.5

    def test_no_sources_returns_defaults(self):
        cfg = Config.resolve()
        assert cfg.to_dict() == DEFAULT_CONFIG

    def test_toml_flat_vs_nested_same_result(self, monkeypatch, tmp_path):
        for k in [ENV_KEY_CONFIDENCE, ENV_KEY_FORMAT, ENV_KEY_DIR, ENV_KEY_RANDOM]:
            monkeypatch.delenv(k, raising=False)
        try:
            import tomli_w
        except ImportError:
            pytest.skip("tomli-w not available")

        nested = tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", dir=str(tmp_path), delete=False)
        tomli_w.dump({"ticket_classifier": {"confidence_threshold": 0.4}}, nested)
        nested.close()

        flat = tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", dir=str(tmp_path), delete=False)
        tomli_w.dump({"confidence_threshold": 0.4}, flat)
        flat.close()

        try:
            c1 = Config.resolve(config_file_path=nested.name)
            c2 = Config.resolve(config_file_path=flat.name)
            assert c1.confidence_threshold == c2.confidence_threshold == 0.4
        finally:
            os.unlink(nested.name)
            os.unlink(flat.name)
