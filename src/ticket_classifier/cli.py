import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .config import Config
from .text_reader import TextReader
from .sentiment_classifier import SentimentClassifier
from .report_generator import ReportGenerator, VALID_OUTPUT_FORMATS


TRAINING_DATA = {
    "positive": [
        "This product is amazing, I love it!",
        "Great service, highly recommended!",
        "Excellent quality, worth every penny.",
        "The best experience I've had so far.",
        "Super happy with my purchase.",
        "Wonderful product, fantastic experience!",
        "Love everything about this, perfect!",
        "Amazing customer support, thank you!",
        "Incredible value for money, buy it!",
        "Absolutely fantastic, 5 stars!",
    ],
    "negative": [
        "This is terrible, very disappointed.",
        "Worst customer service ever.",
        "Horrible quality, don't buy it.",
        "Completely useless, waste of money.",
        "I hate this product so much.",
        "Awful experience, never coming back.",
        "Broken on arrival, total garbage.",
        "Terrible, would not recommend.",
        "Disgusting quality, avoid at all costs.",
        "Worst purchase ever made.",
    ],
    "neutral": [
        "The product is okay, nothing special.",
        "It works as expected, no complaints.",
        "Average quality, could be better.",
        "Not bad, but not great either.",
        "It's fine, meets basic needs.",
        "Decent product, nothing extraordinary.",
        "Mediocre at best, pretty standard.",
        "Acceptable, but I've seen better.",
        "Fair quality, nothing to write home about.",
        "It's alright, does the job.",
    ],
}


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

    reader = TextReader()
    if input_path.is_dir():
        tickets = reader.read_directory(str(input_path))
    elif input_path.is_file():
        content = reader.read_file(str(input_path))
        tickets = [{"id": 0, "file_name": input_path.name, "content": content}]
    else:
        print(f"Error: Invalid input path: {parsed.input_path}", file=sys.stderr)
        return 1

    if not tickets:
        print("Error: No tickets found at the specified path.", file=sys.stderr)
        return 1

    texts = [t["content"] for t in tickets]
    train_texts = []
    train_labels = []
    for label, samples in TRAINING_DATA.items():
        train_texts.extend(samples)
        train_labels.extend([label] * len(samples))

    classifier = SentimentClassifier(random_state=config.random_state)
    classifier.train(train_texts, train_labels)

    predictions = classifier.predict_batch(texts)

    report_gen = ReportGenerator(
        low_confidence_threshold=config.confidence_threshold,
        output_format=config.output_format
    )

    output_dir = config.output_dir
    outputs = report_gen.generate_full_report(
        predictions, tickets, output_dir=output_dir
    )

    for key, value in outputs.items():
        if Path(value).exists():
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: (in-memory, {len(value)} chars)")

    low_conf_count = sum(1 for p in predictions if p["confidence"] < config.confidence_threshold)
    print(f"\nProcessed {len(tickets)} tickets, {low_conf_count} below threshold {config.confidence_threshold}")

    return 0
