import os
import csv
from pathlib import Path
from typing import List, Dict, Union, Optional


class TextReader:
    def __init__(self, encoding: str = "utf-8"):
        self.encoding = encoding

    def read_file(self, file_path: str) -> str:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not path.is_file():
            raise ValueError(f"Not a file: {file_path}")

        suffix = path.suffix.lower()
        if suffix == ".txt":
            return self._read_txt(path)
        elif suffix == ".csv":
            return self._read_csv(path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def read_directory(self, dir_path: str, pattern: str = "*.txt") -> List[Dict[str, Union[str, int]]]:
        path = Path(dir_path)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        if not path.is_dir():
            raise ValueError(f"Not a directory: {dir_path}")

        tickets = []
        for idx, file_path in enumerate(sorted(path.glob(pattern))):
            if file_path.is_file():
                content = self.read_file(str(file_path))
                tickets.append({
                    "id": idx,
                    "file_name": file_path.name,
                    "content": content
                })
        return tickets

    def _read_txt(self, path: Path) -> str:
        with open(path, "r", encoding=self.encoding) as f:
            return f.read().strip()

    def _read_csv(self, path: Path, text_column: Optional[str] = None) -> str:
        texts = []
        with open(path, "r", encoding=self.encoding, newline="") as f:
            reader = csv.DictReader(f)
            if text_column is None:
                text_column = reader.fieldnames[0] if reader.fieldnames else "text"
            for row in reader:
                if text_column in row:
                    texts.append(row[text_column].strip())
        return "\n".join(texts)
