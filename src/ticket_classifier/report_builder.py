from pathlib import Path
from typing import Dict, List, Optional

from .config import Config
from .text_reader import TextReader
from .sentiment_classifier import SentimentClassifier
from .report_generator import ReportGenerator


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


class ReportBuilder:
    def __init__(self, config: Config):
        self._config = config
        self._reader = TextReader()
        self._classifier = SentimentClassifier(random_state=config.random_state)
        self._classifier.train(*self._prepare_training_data())

    def _prepare_training_data(self):
        train_texts = []
        train_labels = []
        for label, samples in TRAINING_DATA.items():
            train_texts.extend(samples)
            train_labels.extend([label] * len(samples))
        return train_texts, train_labels

    def read_tickets(self, input_path: str) -> List[Dict]:
        path = Path(input_path)
        if path.is_dir():
            return self._reader.read_directory(str(path))
        if path.is_file():
            content = self._reader.read_file(str(path))
            return [{"id": 0, "file_name": path.name, "content": content}]
        raise FileNotFoundError(f"Input path not found: {input_path}")

    def predict(self, tickets: List[Dict]) -> List[Dict]:
        texts = [t["content"] for t in tickets]
        return self._classifier.predict_batch(texts)

    def build_report(
        self,
        predictions: List[Dict],
        tickets: List[Dict],
        output_dir: Optional[str] = None,
        output_format: Optional[str] = None
    ) -> Dict[str, str]:
        report_gen = ReportGenerator(
            low_confidence_threshold=self._config.confidence_threshold,
            output_format=output_format or self._config.output_format,
        )
        return report_gen.generate_full_report(
            predictions, tickets,
            output_dir=output_dir or self._config.output_dir,
        )

    def run_pipeline(self, input_path: str) -> Dict:
        tickets = self.read_tickets(input_path)
        if not tickets:
            raise ValueError("No tickets found at the specified path.")
        predictions = self.predict(tickets)
        outputs = self.build_report(predictions, tickets)

        low_conf_count = sum(
            1 for p in predictions
            if p["confidence"] < self._config.confidence_threshold
        )
        outputs["_meta"] = {
            "total_tickets": len(tickets),
            "low_confidence_count": low_conf_count,
            "threshold": self._config.confidence_threshold,
        }
        return outputs
