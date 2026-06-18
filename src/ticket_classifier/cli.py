import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .config import Config
from .report_generator import VALID_OUTPUT_FORMATS
from .report_builder import ReportBuilder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ticket-sentiment",
        description="Ticket Sentiment Classifier - classify ticket text into positive/neutral/negative"
    )

    parser.add_argument(
        "input_path",
        type=str,
        help="Path to a ticket file or directory of ticket files"
    )
    parser.add_argument(
        "-t", "--confidence-threshold",
        type=float,
        default=None,
        help="Confidence threshold for low-confidence detection (0, 1] (default: 0.7)"
    )
    parser.add_argument(
        "-f", "--output-format",
        type=str,
        choices=sorted(VALID_OUTPUT_FORMATS),
        default=None,
        help="Output report format: json, csv, or both (default: both)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Output directory for report files (default: output)"
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Path to config file (.json or .toml)"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: 42)"
    )

    return parser


def load_config(args: argparse.Namespace) -> Config:
    return Config.resolve(cli_args=args, config_file_path=args.config)


def run(args: Optional[List[str]] = None) -> int:
    parser = build_parser()
    parsed = parser.parse_args(args)

    config = load_config(parsed)

    input_path = Path(parsed.input_path)
    if not input_path.exists():
        print(f"Error: Input path not found: {parsed.input_path}", file=sys.stderr)
        return 1

    builder = ReportBuilder(config)

    try:
        result = builder.run_pipeline(parsed.input_path)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    outputs = {k: v for k, v in result.items() if k != "_meta"}
    meta = result.get("_meta", {})

    for key, value in outputs.items():
        if Path(value).exists():
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: (in-memory, {len(value)} chars)")

    total = meta.get("total_tickets", 0)
    low_conf = meta.get("low_confidence_count", 0)
    threshold = meta.get("threshold", 0.7)
    print(f"\nProcessed {total} tickets, {low_conf} below threshold {threshold}")

    return 0
