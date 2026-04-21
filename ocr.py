"""
ocr.py — Unified OCR interface with pluggable backends.

Supports:
  - EasyOCR  (default, 83 languages, better accuracy on screens)
  - Tesseract (legacy fallback, lighter weight)

Both backends produce identical output format so the rest of the pipeline
is completely backend-agnostic.
"""

import hashlib
import logging
import os

import cv2
import numpy as np
import pytesseract

from config import config

logger = logging.getLogger(__name__)


def get_ocr_engine():
    """Factory: return the configured OCR engine instance.

    Checks ``config["ocr_backend"]`` — ``"easyocr"`` or ``"tesseract"``.
    Falls back to Tesseract if EasyOCR is not installed.
    """
    backend = config.get("ocr_backend", "easyocr")

    if backend == "easyocr":
        try:
            from ocr_easyocr import EasyOCREngine
            logger.info("Using EasyOCR backend")
            return EasyOCREngine()
        except (ImportError, RuntimeError) as exc:
            logger.warning(f"EasyOCR unavailable ({exc}), falling back to Tesseract")
            return TesseractEngine()

    return TesseractEngine()


class TesseractEngine:
    """Extract text from images using Tesseract (legacy backend)."""

    def __init__(self):
        self._last_hash = None
        self._last_text = ""
        self._last_boxes_hash = None
        self._last_boxes = []
        self._configure_tesseract()

    def _configure_tesseract(self):
        """Set Tesseract executable path from config."""
        tess_path = config["tesseract_path"]
        if tess_path and os.path.exists(tess_path):
            pytesseract.pytesseract.tesseract_cmd = tess_path

    def _cap_resolution(self, image):
        """Downscale image if it exceeds the configured max width to save CPU."""
        max_w = int(config.get("capture_max_width", 1920))
        h, w = image.shape[:2]
        if w <= max_w:
            return image, 1.0
        scale = max_w / w
        new_w = max_w
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale

    def _preprocess_with_scale(self, image):
        """Preprocess image and keep track of OCR scaling."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape
        scale = 1.0
        if w < 900:
            scale = min(2.0, 900 / max(1, w))
            gray = cv2.resize(
                gray,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_CUBIC,
            )

        # Lightweight denoise
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)

        binary = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )

        return binary, scale

    def preprocess(self, image):
        """Preprocess image for better OCR accuracy."""
        processed, _ = self._preprocess_with_scale(image)
        return processed

    def extract_text(self, image, preprocess=True):
        """Extract text from an image."""
        img_hash = self._hash_image(image)
        if img_hash == self._last_hash:
            return self._last_text

        if preprocess:
            processed, _ = self._preprocess_with_scale(image)
        else:
            processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        try:
            text = pytesseract.image_to_string(
                processed,
                lang=config["ocr_language"],
                config="--psm 6"
            )
        except pytesseract.TesseractNotFoundError:
            return "[ERROR] Tesseract not found."
        except Exception as e:
            return f"[OCR Error] {e}"

        text = self._clean_text(text)
        self._last_hash = img_hash
        self._last_text = text
        return text

    def extract_text_with_boxes(self, image):
        """Extract text with bounding box positions and intelligent filtering.

        Returns:
            list[dict]: List of {text, x, y, w, h, conf} dicts.
        """
        capped_image, res_scale = self._cap_resolution(image)

        img_hash = self._hash_image(capped_image)
        if img_hash == self._last_boxes_hash:
            return list(self._last_boxes)

        processed, ocr_scale = self._preprocess_with_scale(capped_image)
        total_scale = ocr_scale

        try:
            data = pytesseract.image_to_data(
                processed,
                lang=config["ocr_language"],
                config="--oem 3 --psm 11",
                output_type=pytesseract.Output.DICT
            )
        except Exception:
            return []

        results = []
        n = len(data["text"])

        for i in range(n):
            text = data["text"][i].strip()
            conf = self._parse_confidence(data["conf"][i])

            if not text:
                continue
            if conf < 30:
                continue

            text_len = len(text)
            if text_len <= 2 and conf < 60:
                continue
            elif text_len <= 4 and conf < 45:
                continue

            if self._is_likely_noise(text, conf):
                continue

            raw_x = int(round(data["left"][i] / total_scale))
            raw_y = int(round(data["top"][i] / total_scale))
            raw_w = int(round(data["width"][i] / total_scale))
            raw_h = int(round(data["height"][i] / total_scale))

            if res_scale != 1.0:
                raw_x = int(round(raw_x / res_scale))
                raw_y = int(round(raw_y / res_scale))
                raw_w = int(round(raw_w / res_scale))
                raw_h = int(round(raw_h / res_scale))

            results.append({
                "text": text,
                "x": raw_x,
                "y": raw_y,
                "w": raw_w,
                "h": raw_h,
                "conf": conf,
                "ocr_block_id": (
                    data.get("block_num", [0])[i],
                    data.get("par_num", [0])[i],
                ),
                "ocr_line_id": (
                    data.get("block_num", [0])[i],
                    data.get("par_num", [0])[i],
                    data.get("line_num", [0])[i],
                ),
                "ocr_word_index": data.get("word_num", [i])[i],
            })

        self._last_boxes_hash = img_hash
        self._last_boxes = list(results)
        return results

    def _is_likely_noise(self, text, confidence):
        """Determine if OCR result is likely noise."""
        if len(text) == 1 and confidence < 70:
            return True

        special_char_ratio = sum(1 for c in text if not c.isalnum()) / len(text)
        if special_char_ratio > 0.7 and confidence < 60:
            return True

        if len(text) <= 3 and any(c.isdigit() for c in text) and any(not c.isalnum() for c in text):
            return True

        noise_patterns = ['|', 'l', 'i', 'o', '0', 'O']
        if all(c in noise_patterns for c in text) and confidence < 50:
            return True

        return False

    @staticmethod
    def _parse_confidence(value):
        """Safely parse Tesseract confidence values."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return -1.0

    @staticmethod
    def _hash_image(image):
        """Fast perceptual hash of an image for dedup."""
        small = cv2.resize(image, (16, 16))
        return hashlib.md5(small.tobytes()).hexdigest()

    @staticmethod
    def _clean_text(text):
        """Clean OCR output."""
        lines = text.strip().split("\n")
        cleaned = []
        for line in lines:
            line = line.strip()
            if len(line) > 1 and any(c.isalnum() for c in line):
                cleaned.append(line)
        return "\n".join(cleaned)


# Default engine — created once on import via factory
_default_engine = None


def _get_default_engine():
    global _default_engine
    if _default_engine is None:
        _default_engine = get_ocr_engine()
    return _default_engine


class OCREngine:
    """Unified adapter that delegates to the configured backend.

    This is the class that the rest of the codebase imports.
    It transparently forwards calls to EasyOCR or Tesseract.
    """

    def __init__(self):
        self._engine = get_ocr_engine()

    def extract_text_with_boxes(self, image):
        return self._engine.extract_text_with_boxes(image)

    def extract_text(self, image, preprocess=True):
        if hasattr(self._engine, "extract_text"):
            return self._engine.extract_text(image, preprocess=preprocess)
        # EasyOCR path — concatenate box texts
        boxes = self._engine.extract_text_with_boxes(image)
        return "\n".join(b["text"] for b in boxes)

    def preprocess(self, image):
        if hasattr(self._engine, "preprocess"):
            return self._engine.preprocess(image)
        return image


# Module-level convenience instance
ocr_engine = OCREngine()
