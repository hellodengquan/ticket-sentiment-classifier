import json
import csv
from collections import Counter
from pathlib import Path
from typing import List, Dict, Union, Optional
from datetime import datetime


class ReportGenerator:
    def __init__(self, low_confidence_threshold: float = 0.7):
        self.low_confidence_threshold = low_confidence_threshold

    def generate_batch_summary(
        self,
        predictions: List[Dict[str, Union[str, float]]],
        tickets: Optional[List[Dict[str, Union[str, int]]]] = None
    ) -> Dict[str, Union[int, float, Dict[str, int], List[Dict]]]:
        if not predictions:
            raise ValueError("No predictions provided")

        total = len(predictions)
        sentiments = [p["sentiment"] for p in predictions]
        sentiment_counts = Counter(sentiments)

        confidences = [float(p["confidence"]) for p in predictions]
        avg_confidence = sum(confidences) / total
        min_confidence = min(confidences)
        max_confidence = max(confidences)

        low_confidence = [
            {
                "index": idx,
                "file_name": tickets[idx]["file_name"] if tickets else f"ticket_{idx}",
                "sentiment": p["sentiment"],
                "confidence": p["confidence"],
                "probabilities": p["probabilities"]
            }
            for idx, p in enumerate(predictions)
            if p["confidence"] < self.low_confidence_threshold
        ]

        summary = {
            "total_tickets": total,
            "sentiment_distribution": dict(sentiment_counts),
            "sentiment_percentages": {
                label: round((count / total) * 100, 2)
                for label, count in sentiment_counts.items()
            },
            "confidence_stats": {
                "average": round(avg_confidence, 4),
                "min": round(min_confidence, 4),
                "max": round(max_confidence, 4)
            },
            "low_confidence_count": len(low_confidence),
            "low_confidence_percentage": round((len(low_confidence) / total) * 100, 2),
            "low_confidence_samples": low_confidence,
            "generated_at": datetime.now().isoformat()
        }

        return summary

    def generate_low_confidence_report(
        self,
        low_confidence_samples: List[Dict],
        output_path: Optional[str] = None
    ) -> str:
        report = {
            "report_type": "low_confidence_tickets",
            "threshold": self.low_confidence_threshold,
            "total_samples": len(low_confidence_samples),
            "generated_at": datetime.now().isoformat(),
            "samples": low_confidence_samples
        }

        if output_path:
            self._write_json(report, output_path)
            return output_path
        return json.dumps(report, indent=2, ensure_ascii=False)

    def generate_full_report(
        self,
        predictions: List[Dict[str, Union[str, float]]],
        tickets: Optional[List[Dict[str, Union[str, int]]]] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, str]:
        batch_summary = self.generate_batch_summary(predictions, tickets)
        low_conf_samples = batch_summary["low_confidence_samples"]

        outputs = {}

        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            summary_path = str(Path(output_dir) / "batch_summary.json")
            low_conf_path = str(Path(output_dir) / "low_confidence_report.json")
            csv_path = str(Path(output_dir) / "predictions.csv")

            self._write_json(batch_summary, summary_path)
            self._write_json(
                {
                    "report_type": "low_confidence_tickets",
                    "threshold": self.low_confidence_threshold,
                    "total_samples": len(low_conf_samples),
                    "generated_at": datetime.now().isoformat(),
                    "samples": low_conf_samples
                },
                low_conf_path
            )
            self._write_csv(predictions, tickets, csv_path)

            outputs = {
                "batch_summary": summary_path,
                "low_confidence_report": low_conf_path,
                "predictions_csv": csv_path
            }
        else:
            outputs = {
                "batch_summary": json.dumps(batch_summary, indent=2, ensure_ascii=False),
                "low_confidence_report": self.generate_low_confidence_report(low_conf_samples),
                "predictions_csv": self._predictions_to_csv(predictions, tickets)
            }

        return outputs

    def _write_json(self, data: Dict, file_path: str) -> None:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _write_csv(
        self,
        predictions: List[Dict[str, Union[str, float]]],
        tickets: Optional[List[Dict[str, Union[str, int]]]],
        file_path: str
    ) -> None:
        headers = ["id", "file_name", "sentiment", "confidence"]
        if predictions and "probabilities" in predictions[0]:
            prob_labels = sorted(predictions[0]["probabilities"].keys())
            headers.extend([f"prob_{label}" for label in prob_labels])

        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

            for idx, pred in enumerate(predictions):
                row = {
                    "id": idx,
                    "file_name": tickets[idx]["file_name"] if tickets else f"ticket_{idx}",
                    "sentiment": pred["sentiment"],
                    "confidence": pred["confidence"]
                }
                if "probabilities" in pred:
                    for label in prob_labels:
                        row[f"prob_{label}"] = round(pred["probabilities"][label], 4)
                writer.writerow(row)

    def _predictions_to_csv(
        self,
        predictions: List[Dict[str, Union[str, float]]],
        tickets: Optional[List[Dict[str, Union[str, int]]]]
    ) -> str:
        import io
        output = io.StringIO()

        headers = ["id", "file_name", "sentiment", "confidence"]
        if predictions and "probabilities" in predictions[0]:
            prob_labels = sorted(predictions[0]["probabilities"].keys())
            headers.extend([f"prob_{label}" for label in prob_labels])

        writer = csv.DictWriter(output, fieldnames=headers)
        writer.writeheader()

        for idx, pred in enumerate(predictions):
            row = {
                "id": idx,
                "file_name": tickets[idx]["file_name"] if tickets else f"ticket_{idx}",
                "sentiment": pred["sentiment"],
                "confidence": pred["confidence"]
            }
            if "probabilities" in pred:
                for label in prob_labels:
                    row[f"prob_{label}"] = round(pred["probabilities"][label], 4)
            writer.writerow(row)

        return output.getvalue()
