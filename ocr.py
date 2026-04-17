"""
ocr.py — Text extraction from screen captures using Tesseract OCR.

Includes image preprocessing for better accuracy and text deduplication
to avoid re-processing identical frames.
"""

import hashlib
import os

import cv2
import numpy as np
import pytesseract

from config import config


class OCREngine:
    """Extract text from images using Tesseract with preprocessing."""

    def __init__(self):
        self._last_hash = None
        self._last_text = ""
        self._configure_tesseract()

    def _configure_tesseract(self):
        """Set Tesseract executable path from config."""
        tess_path = config["tesseract_path"]
        if tess_path and os.path.exists(tess_path):
            pytesseract.pytesseract.tesseract_cmd = tess_path

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
        # Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Upscale if image is small (helps with tiny text)
        h, w = gray.shape
        if w < 800:
            scale = 2
            gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

        # Denoise while preserving edges
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)

        # Adaptive thresholding for varied lighting
        binary = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )

        return binary

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
            processed = self.preprocess(image)
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
        """Extract text with bounding box positions.

        Returns:
            list[dict]: List of {text, x, y, w, h, conf} dicts.
        """
        processed = self.preprocess(image)

        try:
            data = pytesseract.image_to_data(
                processed,
                lang=config["ocr_language"],
                output_type=pytesseract.Output.DICT
            )
        except Exception:
            return []

        results = []
        n = len(data["text"])
        for i in range(n):
            text = data["text"][i].strip()
            conf = int(data["conf"][i])
            if text and conf > 40:  # confidence threshold
                results.append({
                    "text": text,
                    "x": data["left"][i],
                    "y": data["top"][i],
                    "w": data["width"][i],
                    "h": data["height"][i],
                    "conf": conf,
                })

        return results

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
