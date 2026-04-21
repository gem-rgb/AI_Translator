"""
ocr_easyocr.py — EasyOCR backend for multilingual screen text extraction.

EasyOCR supports 83 languages, handles screen fonts better than Tesseract,
and produces more reliable bounding boxes.  Runs on CPU by default.
"""

import hashlib
import logging

import cv2
import numpy as np

from config import config

logger = logging.getLogger(__name__)

# Lazy-loaded global reader to avoid re-initializing on every call
_reader = None
_reader_languages = None


def _get_reader(languages=None):
    """Get or create the EasyOCR reader (lazy, singleton).

    EasyOCR models are downloaded once (~100-300MB per language pack) and
    cached globally in ``~/.EasyOCR/``.
    """
    global _reader, _reader_languages

    if languages is None:
        languages = config.get("easyocr_languages", ["en"])

    # Normalize to sorted tuple for comparison
    lang_key = tuple(sorted(languages))

    if _reader is not None and _reader_languages == lang_key:
        return _reader

    try:
        import easyocr
    except ImportError:
        raise RuntimeError(
            "EasyOCR is not installed.  Run:  pip install easyocr\n"
            "Note: this also installs PyTorch (~200MB CPU-only)."
        )

    use_gpu = config.get("easyocr_gpu", False)
    logger.info(
        f"Initialising EasyOCR reader: languages={list(lang_key)}, gpu={use_gpu}"
    )

    _reader = easyocr.Reader(
        list(lang_key),
        gpu=use_gpu,
        verbose=False,
    )
    _reader_languages = lang_key
    return _reader


class EasyOCREngine:
    """Extract text from images using EasyOCR."""

    def __init__(self):
        self._last_boxes_hash = None
        self._last_boxes = []

    def extract_text_with_boxes(self, image):
        """Extract text with bounding-box positions.

        Returns the same dict format as ``OCREngine.extract_text_with_boxes``
        so the rest of the pipeline is backend-agnostic:

            list[dict]:  {text, x, y, w, h, conf, ocr_block_id, ocr_line_id, ocr_word_index}
        """
        img_hash = self._hash_image(image)
        if img_hash == self._last_boxes_hash:
            return list(self._last_boxes)

        # Cap resolution for CPU performance — EasyOCR is heavier than Tesseract
        max_w = int(config.get("easyocr_max_width", 1280))
        resized, res_scale = self._cap_resolution(image, max_w)

        reader = _get_reader()

        try:
            # Convert BGR → RGB for EasyOCR
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            raw_results = reader.readtext(
                rgb,
                paragraph=False,
                min_size=10,
                text_threshold=0.5,
                low_text=0.4,
                width_ths=0.7,
            )
        except Exception as exc:
            logger.error(f"EasyOCR error: {exc}")
            return []

        results = []
        block_id = 0

        for bbox, text, conf in raw_results:
            text = text.strip()
            if not text:
                continue
            if conf < 0.25:
                continue

            # bbox is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] (polygon)
            xs = [int(p[0]) for p in bbox]
            ys = [int(p[1]) for p in bbox]
            x = min(xs)
            y = min(ys)
            w = max(xs) - x
            h = max(ys) - y

            # Map back to original image coordinates
            if res_scale != 1.0:
                x = int(round(x / res_scale))
                y = int(round(y / res_scale))
                w = int(round(w / res_scale))
                h = int(round(h / res_scale))

            if w < 4 or h < 4:
                continue

            block_id += 1
            results.append({
                "text": text,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "conf": conf * 100,  # normalise to 0-100 like Tesseract
                "ocr_block_id": (block_id, 1),
                "ocr_line_id": (block_id, 1, 1),
                "ocr_word_index": 1,
            })

        self._last_boxes_hash = img_hash
        self._last_boxes = list(results)
        return results

    @staticmethod
    def _cap_resolution(image, max_width):
        """Downscale if wider than max_width — critical for CPU performance."""
        h, w = image.shape[:2]
        if w <= max_width:
            return image, 1.0
        scale = max_width / w
        new_w = max_width
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale

    @staticmethod
    def _hash_image(image):
        """Fast perceptual hash for dedup."""
        small = cv2.resize(image, (16, 16))
        return hashlib.md5(small.tobytes()).hexdigest()
