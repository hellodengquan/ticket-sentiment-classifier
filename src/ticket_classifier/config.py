import json
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_CONFIG = {
    "confidence_threshold": 0.7,
    "output_format": "both",
    "output_dir": "output",
    "random_state": 42,
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
    def from_file(cls, file_path: str) -> "Config":
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data)

    def to_file(self, file_path: str) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

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
