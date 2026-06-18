import numpy as np
from typing import List, Dict, Tuple, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class SentimentClassifier:
    SENTIMENT_LABELS = ["negative", "neutral", "positive"]

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self._is_trained = False
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            min_df=2,
            stop_words="english"
        )
        self.classifier = LogisticRegression(
            C=5.0,
            max_iter=1000,
            random_state=self.random_state
        )
        self.model = Pipeline([
            ("tfidf", self.vectorizer),
            ("clf", self.classifier)
        ])

    def train(self, texts: List[str], labels: List[str]) -> Dict[str, float]:
        if len(texts) != len(labels):
            raise ValueError("Number of texts and labels must match")
        if len(texts) < 10:
            raise ValueError("Need at least 10 samples for training")

        label_set = set(self.SENTIMENT_LABELS)
        for label in labels:
            if label not in label_set:
                raise ValueError(f"Invalid label '{label}'. Must be one of {self.SENTIMENT_LABELS}")

        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=self.random_state, stratify=labels
        )

        self.model.fit(X_train, y_train)
        self._is_trained = True

        y_pred = self.model.predict(X_test)
        return {
            "train_size": len(X_train),
            "test_size": len(X_test),
            "accuracy": accuracy_score(y_test, y_pred)
        }

    def predict(self, text: str) -> Dict[str, Union[str, float]]:
        if not self._is_trained:
            raise RuntimeError("Model is not trained. Call train() first.")

        prediction = self.model.predict([text])[0]
        probabilities = self.model.predict_proba([text])[0]
        confidence = float(np.max(probabilities))

        return {
            "sentiment": prediction,
            "confidence": confidence,
            "probabilities": {
                label: float(prob)
                for label, prob in zip(self.model.classes_, probabilities)
            }
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Union[str, float]]]:
        return [self.predict(text) for text in texts]

    def get_low_confidence_samples(
        self,
        texts: List[str],
        threshold: float = 0.7
    ) -> List[Dict[str, Union[int, str, float]]]:
        results = []
        for idx, text in enumerate(texts):
            pred = self.predict(text)
            if pred["confidence"] < threshold:
                results.append({
                    "index": idx,
                    "text": text,
                    "sentiment": pred["sentiment"],
                    "confidence": pred["confidence"],
                    "probabilities": pred["probabilities"]
                })
        return results

    @property
    def is_trained(self) -> bool:
        return self._is_trained
