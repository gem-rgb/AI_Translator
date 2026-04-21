"""
ui_classifier.py — Runtime UI element classifier.

Loads the trained sklearn model (from ``models/ui_classifier.pkl``) and
provides a fast ``predict()`` call that replaces heuristic UI detection.

Falls back to heuristic scoring if the model file is not found.
"""

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "ui_classifier.pkl",
)


class UIClassifier:
    """Binary classifier: UI element (1) vs translatable content (0).

    Loads a serialised sklearn pipeline on first use and caches it.
    If the model file doesn't exist, ``predict()`` returns ``None``
    and the caller should fall back to heuristic scoring.
    """

    def __init__(self):
        self._model = None
        self._available = None  # None = not checked yet

    @property
    def available(self):
        """Check if a trained model exists."""
        if self._available is None:
            self._available = os.path.exists(MODEL_PATH)
        return self._available

    def _load(self):
        """Load the model on first call."""
        if self._model is not None:
            return True

        if not self.available:
            return False

        try:
            import joblib
            self._model = joblib.load(MODEL_PATH)
            logger.info(f"Loaded UI classifier from {MODEL_PATH}")
            return True
        except Exception as exc:
            logger.warning(f"Failed to load UI classifier: {exc}")
            self._available = False
            return False

    def predict(self, block, image_w, image_h):
        """Predict whether a text block is a UI element or content.

        Args:
            block: dict with keys: x, y, w, h, text (or compact_text),
                   line_count or lines.
            image_w: image width (for normalisation).
            image_h: image height (for normalisation).

        Returns:
            dict with:
                is_ui: bool — True if UI element, False if content
                confidence: float — model confidence (0-1)
                method: str — "model" or "unavailable"
            Or None if model is not available.
        """
        if not self._load():
            return None

        features = self._extract_features(block, image_w, image_h)
        features_array = np.array([features], dtype=np.float32)

        try:
            prediction = self._model.predict(features_array)[0]
            probabilities = self._model.predict_proba(features_array)[0]
            confidence = float(max(probabilities))

            return {
                "is_ui": bool(prediction == 1),
                "confidence": confidence,
                "method": "model",
            }
        except Exception as exc:
            logger.debug(f"Prediction failed: {exc}")
            return None

    def predict_batch(self, blocks, image_w, image_h):
        """Predict for multiple blocks at once (faster than individual calls).

        Returns:
            list[dict] or None if model unavailable.
        """
        if not self._load():
            return None

        if not blocks:
            return []

        features_list = [
            self._extract_features(b, image_w, image_h) for b in blocks
        ]
        features_array = np.array(features_list, dtype=np.float32)

        try:
            predictions = self._model.predict(features_array)
            probabilities = self._model.predict_proba(features_array)

            results = []
            for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
                results.append({
                    "is_ui": bool(pred == 1),
                    "confidence": float(max(probs)),
                    "method": "model",
                })
            return results
        except Exception as exc:
            logger.debug(f"Batch prediction failed: {exc}")
            return None

    @staticmethod
    def _extract_features(block, image_w, image_h):
        """Extract the same feature vector used during training.

        Feature order must match ``FEATURE_COLUMNS`` in
        ``training/train_ui_classifier.py``.
        """
        import re

        text = block.get("compact_text", block.get("text", "")).strip()
        text = re.sub(r"\s+", " ", text)

        x = block.get("x", 0)
        y = block.get("y", 0)
        w = block.get("w", 0)
        h = block.get("h", 0)

        iw = max(1, image_w)
        ih = max(1, image_h)

        lines = block.get("lines", [])
        line_count = block.get("line_count", len(lines) if lines else 1)

        text_length = len(text)
        word_count = len(text.split()) if text else 0
        is_single_line = 1 if line_count <= 1 else 0
        has_punctuation = 1 if any(c in text for c in ".!?,;:") else 0
        letter_chars = sum(1 for c in text if c.isalpha())
        letter_ratio = letter_chars / max(1, text_length)

        return [
            x / iw,          # x_norm
            y / ih,          # y_norm
            w / iw,          # w_norm
            h / ih,          # h_norm
            text_length,     # text_length
            word_count,      # word_count
            is_single_line,  # is_single_line
            has_punctuation, # has_punctuation
            round(letter_ratio, 4),  # letter_ratio
        ]


# Module-level singleton
ui_classifier = UIClassifier()
