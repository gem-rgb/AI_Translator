"""
ocr.py — Text extraction from screen captures using Tesseract OCR.

Includes image preprocessing for better accuracy and text deduplication
to avoid re-processing identical frames.
"""

import hashlib
import os

import cv2
import pytesseract

from config import config


class OCREngine:
    """Extract text from images using Tesseract with preprocessing."""

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

    def _preprocess_with_scale(self, image):
        """Preprocess image and keep track of OCR scaling."""
        # Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Upscale smaller captures for OCR, but remember the scale so we can
        # map Tesseract boxes back onto the original screenshot accurately.
        h, w = gray.shape
        scale = 1.0
        if w < 900:
            scale = min(2.0, 900 / max(1, w))
            gray = cv2.resize(
                gray,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_CUBIC,
            )

        # Denoise while preserving edges
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)

        # Adaptive thresholding for varied lighting
        binary = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )

        return binary, scale

    def preprocess(self, image):
        """Preprocess image for better OCR accuracy.

        Pipeline:
          1. Convert to grayscale
          2. Resize (upscale small text)
          3. Apply bilateral filter (denoise while keeping edges)
          4. Adaptive threshold for binarization

        Args:
            image: numpy.ndarray in BGR format.

        Returns:
            numpy.ndarray: Preprocessed binary image.
        """
        processed, _ = self._preprocess_with_scale(image)
        return processed

    def extract_text(self, image, preprocess=True):
        """Extract text from an image.

        Args:
            image: numpy.ndarray in BGR format.
            preprocess: Whether to apply preprocessing pipeline.

        Returns:
            str: Extracted text, or empty string if image is unchanged.
        """
        # Check if image has changed (dedup)
        img_hash = self._hash_image(image)
        if img_hash == self._last_hash:
            return self._last_text

        if preprocess:
            processed, _ = self._preprocess_with_scale(image)
        else:
            processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Run Tesseract
        try:
            text = pytesseract.image_to_string(
                processed,
                lang=config["ocr_language"],
                config="--psm 6"  # Assume uniform block of text
            )
        except pytesseract.TesseractNotFoundError:
            return "[ERROR] Tesseract not found. Install from: https://github.com/UB-Mannheim/tesseract/wiki"
        except Exception as e:
            return f"[OCR Error] {e}"

        # Clean up text
        text = self._clean_text(text)

        # Cache result
        self._last_hash = img_hash
        self._last_text = text

        return text

    def extract_text_with_boxes(self, image):
        """Extract text with bounding box positions and intelligent filtering.

        Returns:
            list[dict]: List of {text, x, y, w, h, conf} dicts.
        """
        img_hash = self._hash_image(image)
        if img_hash == self._last_boxes_hash:
            return list(self._last_boxes)

        processed, scale = self._preprocess_with_scale(image)

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

        # Dynamic confidence threshold based on text characteristics
        for i in range(n):
            text = data["text"][i].strip()
            conf = self._parse_confidence(data["conf"][i])

            if not text:
                continue

            # Skip very low confidence results
            if conf < 30:
                continue

            # Apply stricter confidence for short text (likely noise)
            text_len = len(text)
            if text_len <= 2 and conf < 60:
                continue
            elif text_len <= 4 and conf < 45:
                continue

            # Filter out common OCR noise patterns
            if self._is_likely_noise(text, conf):
                continue

            results.append({
                "text": text,
                "x": int(round(data["left"][i] / scale)),
                "y": int(round(data["top"][i] / scale)),
                "w": int(round(data["width"][i] / scale)),
                "h": int(round(data["height"][i] / scale)),
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
        """Determine if OCR result is likely noise based on text patterns and confidence."""

        # Single characters with low confidence are likely noise
        if len(text) == 1 and confidence < 70:
            return True

        # Text with mostly special characters
        special_char_ratio = sum(1 for c in text if not c.isalnum()) / len(text)
        if special_char_ratio > 0.7 and confidence < 60:
            return True

        # Very short text with numbers and symbols mixed
        if len(text) <= 3 and any(c.isdigit() for c in text) and any(not c.isalnum() for c in text):
            return True

        # Common OCR artifacts
        noise_patterns = ['|', 'l', 'i', 'o', '0', 'O']  # Characters often confused
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
        # Downsample to tiny size for fast comparison
        small = cv2.resize(image, (16, 16))
        return hashlib.md5(small.tobytes()).hexdigest()

    @staticmethod
    def _clean_text(text):
        """Clean OCR output: remove excessive whitespace and noise."""
        lines = text.strip().split("\n")
        # Remove empty lines and lines with only special chars
        cleaned = []
        for line in lines:
            line = line.strip()
            # Skip lines that are just noise (single chars, symbols)
            if len(line) > 1 and any(c.isalnum() for c in line):
                cleaned.append(line)
        return "\n".join(cleaned)


# Module-level convenience instance
ocr_engine = OCREngine()
