import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

try:
    if sys.version_info >= (3, 11):
        import tomllib as _toml
    else:
        import tomli as _toml
    _TOML_AVAILABLE = True
except ImportError:
    _TOML_AVAILABLE = False


DEFAULT_CONFIG = {
    "confidence_threshold": 0.7,
    "output_format": "both",
    "output_dir": "output",
    "random_state": 42,
}

ENV_PREFIX = "TICKET_CLASSIFIER_"

ENV_KEY_MAP = {
    f"{ENV_PREFIX}CONFIDENCE_THRESHOLD": "confidence_threshold",
    f"{ENV_PREFIX}OUTPUT_FORMAT": "output_format",
    f"{ENV_PREFIX}OUTPUT_DIR": "output_dir",
    f"{ENV_PREFIX}RANDOM_STATE": "random_state",
}


class Config:
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self._data = dict(DEFAULT_CONFIG)
        if config_dict:
            self._data.update(config_dict)

    @property
    def confidence_threshold(self) -> float:
        return float(self._data["confidence_threshold"])

    @confidence_threshold.setter
    def confidence_threshold(self, value: float) -> None:
        if not 0.0 < value <= 1.0:
            raise ValueError(f"confidence_threshold must be in (0, 1], got {value}")
        self._data["confidence_threshold"] = value

    @property
    def output_format(self) -> str:
        return self._data["output_format"]

    @output_format.setter
    def output_format(self, value: str) -> None:
        valid = {"json", "csv", "both"}
        if value not in valid:
            raise ValueError(f"output_format must be one of {valid}, got '{value}'")
        self._data["output_format"] = value

    @property
    def output_dir(self) -> str:
        return self._data["output_dir"]

    @output_dir.setter
    def output_dir(self, value: str) -> None:
        self._data["output_dir"] = value

    @property
    def random_state(self) -> int:
        return int(self._data["random_state"])

    @random_state.setter
    def random_state(self, value: int) -> None:
        self._data["random_state"] = value

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)

    @classmethod
    def from_env(cls) -> "Config":
        config_dict: Dict[str, Any] = {}
        for env_key, config_key in ENV_KEY_MAP.items():
            if env_key in os.environ:
                raw_value = os.environ[env_key]
                config_dict[config_key] = cls._convert_env_value(config_key, raw_value)
        return cls(config_dict)

    @classmethod
    def _convert_env_value(cls, config_key: str, raw_value: str) -> Any:
        if config_key == "confidence_threshold":
            try:
                return float(raw_value)
            except ValueError:
                raise ValueError(
                    f"Environment variable for {config_key} must be a valid float, got '{raw_value}'"
                )
        if config_key == "random_state":
            try:
                return int(raw_value)
            except ValueError:
                raise ValueError(
                    f"Environment variable for {config_key} must be a valid integer, got '{raw_value}'"
                )
        return raw_value

    @classmethod
    def from_file(cls, file_path: str) -> "Config":
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")

        suffix = path.suffix.lower()
        if suffix in (".json",):
            return cls._from_json(path)
        elif suffix in (".toml",):
            return cls._from_toml(path)
        else:
            raise ValueError(
                f"Unsupported config file format: {suffix}. Use .json or .toml"
            )

    @classmethod
    def _from_json(cls, path: Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data)

    @classmethod
    def _from_toml(cls, path: Path) -> "Config":
        if not _TOML_AVAILABLE:
            raise ImportError(
                "TOML support requires the 'tomli' package for Python < 3.11. "
                "Install it with: pip install tomli"
            )
        with open(path, "rb") as f:
            data = _toml.load(f)
        root = data.get("ticket_classifier", data)
        return cls(root)

    def to_file(self, file_path: str) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix.lower()
        if suffix in (".json",):
            self._to_json(path)
        elif suffix in (".toml",):
            self._to_toml(path)
        else:
            raise ValueError(
                f"Unsupported config file format: {suffix}. Use .json or .toml"
            )

    def _to_json(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    def _to_toml(self, path: Path) -> None:
        import tomli_w

        root = {"ticket_classifier": self._data}
        with open(path, "wb") as f:
            tomli_w.dump(root, f)

    @classmethod
    def from_cli_args(cls, args: Any) -> "Config":
        config_dict = {}
        if hasattr(args, "confidence_threshold") and args.confidence_threshold is not None:
            config_dict["confidence_threshold"] = args.confidence_threshold
        if hasattr(args, "output_format") and args.output_format is not None:
            config_dict["output_format"] = args.output_format
        if hasattr(args, "output_dir") and args.output_dir is not None:
            config_dict["output_dir"] = args.output_dir
        if hasattr(args, "random_state") and args.random_state is not None:
            config_dict["random_state"] = args.random_state
        return cls(config_dict)

    @classmethod
    def resolve(
        cls,
        cli_args: Optional[Any] = None,
        config_file_path: Optional[str] = None
    ) -> "Config":
        env_config = cls.from_env()

        if config_file_path:
            file_config = cls.from_file(config_file_path)
            merged = dict(file_config._data)
            for k, v in env_config._data.items():
                if env_config._data[k] != DEFAULT_CONFIG[k] and merged[k] == DEFAULT_CONFIG[k]:
                    merged[k] = v
        else:
            merged = dict(env_config._data)

        resolved = cls(merged)

        if cli_args is not None:
            if hasattr(cli_args, "confidence_threshold") and cli_args.confidence_threshold is not None:
                resolved.confidence_threshold = cli_args.confidence_threshold
            if hasattr(cli_args, "output_format") and cli_args.output_format is not None:
                resolved.output_format = cli_args.output_format
            if hasattr(cli_args, "output_dir") and cli_args.output_dir is not None:
                resolved.output_dir = cli_args.output_dir
            if hasattr(cli_args, "random_state") and cli_args.random_state is not None:
                resolved.random_state = cli_args.random_state

        return resolved
