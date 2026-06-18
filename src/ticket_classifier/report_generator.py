import json
import csv
import io
from collections import Counter
from pathlib import Path
from typing import List, Dict, Union, Optional
from datetime import datetime


VALID_OUTPUT_FORMATS = {"json", "csv", "both"}


class ReportGenerator:
    def __init__(
        self,
        low_confidence_threshold: float = 0.7,
        output_format: str = "both"
    ):
        self.low_confidence_threshold = low_confidence_threshold
        self.output_format = output_format
        self._validate_output_format(output_format)

    def _validate_output_format(self, fmt: str) -> None:
        if fmt not in VALID_OUTPUT_FORMATS:
            raise ValueError(
                f"output_format must be one of {VALID_OUTPUT_FORMATS}, got '{fmt}'"
            )

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
        output_dir: Optional[str] = None,
        output_format: Optional[str] = None
    ) -> Dict[str, str]:
        fmt = output_format or self.output_format
        self._validate_output_format(fmt)

        batch_summary = self.generate_batch_summary(predictions, tickets)
        low_conf_samples = batch_summary["low_confidence_samples"]

        outputs = {}

        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            outputs = self._write_reports_to_dir(
                fmt, batch_summary, low_conf_samples, predictions, tickets, output_dir
            )
        else:
            outputs = self._build_reports_in_memory(
                fmt, batch_summary, low_conf_samples, predictions, tickets
            )

        return outputs

    def _write_reports_to_dir(
        self,
        fmt: str,
        batch_summary: Dict,
        low_conf_samples: List[Dict],
        predictions: List[Dict],
        tickets: Optional[List[Dict]],
        output_dir: str
    ) -> Dict[str, str]:
        outputs = {}
        low_conf_report = {
            "report_type": "low_confidence_tickets",
            "threshold": self.low_confidence_threshold,
            "total_samples": len(low_conf_samples),
            "generated_at": datetime.now().isoformat(),
            "samples": low_conf_samples
        }

        if fmt in ("json", "both"):
            summary_path = str(Path(output_dir) / "batch_summary.json")
            low_conf_path = str(Path(output_dir) / "low_confidence_report.json")
            self._write_json(batch_summary, summary_path)
            self._write_json(low_conf_report, low_conf_path)
            outputs["batch_summary"] = summary_path
            outputs["low_confidence_report"] = low_conf_path

        if fmt in ("csv", "both"):
            predictions_csv_path = str(Path(output_dir) / "predictions.csv")
            low_conf_csv_path = str(Path(output_dir) / "low_confidence.csv")
            self._write_csv(predictions, tickets, predictions_csv_path)
            self._write_low_confidence_csv(low_conf_samples, low_conf_csv_path)
            outputs["predictions_csv"] = predictions_csv_path
            outputs["low_confidence_csv"] = low_conf_csv_path

        return outputs

    def _build_reports_in_memory(
        self,
        fmt: str,
        batch_summary: Dict,
        low_conf_samples: List[Dict],
        predictions: List[Dict],
        tickets: Optional[List[Dict]]
    ) -> Dict[str, str]:
        outputs = {}
        low_conf_report = {
            "report_type": "low_confidence_tickets",
            "threshold": self.low_confidence_threshold,
            "total_samples": len(low_conf_samples),
            "generated_at": datetime.now().isoformat(),
            "samples": low_conf_samples
        }

        if fmt in ("json", "both"):
            outputs["batch_summary"] = json.dumps(batch_summary, indent=2, ensure_ascii=False)
            outputs["low_confidence_report"] = json.dumps(low_conf_report, indent=2, ensure_ascii=False)

        if fmt in ("csv", "both"):
            outputs["predictions_csv"] = self._predictions_to_csv(predictions, tickets)
            outputs["low_confidence_csv"] = self._low_confidence_to_csv(low_conf_samples)

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
        content = self._predictions_to_csv(predictions, tickets)
        with open(file_path, "w", encoding="utf-8", newline="") as f:
            f.write(content)

    def _write_low_confidence_csv(
        self,
        low_conf_samples: List[Dict],
        file_path: str
    ) -> None:
        content = self._low_confidence_to_csv(low_conf_samples)
        with open(file_path, "w", encoding="utf-8", newline="") as f:
            f.write(content)

    def _predictions_to_csv(
        self,
        predictions: List[Dict[str, Union[str, float]]],
        tickets: Optional[List[Dict[str, Union[str, int]]]]
    ) -> str:
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

    def _low_confidence_to_csv(
        self,
        low_conf_samples: List[Dict]
    ) -> str:
        output = io.StringIO()

        headers = ["index", "file_name", "sentiment", "confidence"]
        if low_conf_samples and "probabilities" in low_conf_samples[0]:
            prob_labels = sorted(low_conf_samples[0]["probabilities"].keys())
            headers.extend([f"prob_{label}" for label in prob_labels])

        writer = csv.DictWriter(output, fieldnames=headers)
        writer.writeheader()

        for sample in low_conf_samples:
            row = {
                "index": sample.get("index", ""),
                "file_name": sample.get("file_name", ""),
                "sentiment": sample.get("sentiment", ""),
                "confidence": sample.get("confidence", "")
            }
            if "probabilities" in sample:
                for label in prob_labels:
                    row[f"prob_{label}"] = round(sample["probabilities"][label], 4)
            writer.writerow(row)

        return output.getvalue()
