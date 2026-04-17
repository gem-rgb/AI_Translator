"""
translator.py — Language detection and translation engine.

Supports:
  - Online mode: Google Translate via deep-translator (free, no API key)
  - Offline mode: HuggingFace MarianMT models (requires model download)
  - LRU translation cache to avoid redundant API calls
"""

import functools
import logging

from langdetect import detect, LangDetectException

from config import config

logger = logging.getLogger(__name__)


class TranslationCache:
    """LRU cache wrapper for translations keyed by (text, source, target)."""

    def __init__(self, maxsize=256):
        self._maxsize = maxsize
        self._cache = {}
        self._order = []

    def get(self, key):
        """Get cached translation or None."""
        return self._cache.get(key)

    def put(self, key, value):
        """Store a translation in the cache."""
        if key in self._cache:
            self._order.remove(key)
        elif len(self._cache) >= self._maxsize:
            oldest = self._order.pop(0)
            del self._cache[oldest]
        self._cache[key] = value
        self._order.append(key)

    def clear(self):
        self._cache.clear()
        self._order.clear()

    @property
    def size(self):
        return len(self._cache)


class Translator:
    """Main translation engine with language detection and caching."""

    def __init__(self):
        self._cache = TranslationCache(maxsize=config["cache_size"])
        self._online_translator = None
        self._offline_model = None
        self._offline_tokenizer = None

    def detect_language(self, text):
        """Detect the language of the given text.

        Args:
            text: Input text string.

        Returns:
            str: ISO 639-1 language code (e.g., 'en', 'fr', 'ar') or 'unknown'.
        """
        if not text or len(text.strip()) < 3:
            return "unknown"

        try:
            lang = detect(text)
            return lang
        except LangDetectException:
            return "unknown"

    def needs_translation(self, text):
        """Check if text needs translation (i.e., is not in the target language).

        Args:
            text: Input text.

        Returns:
            tuple: (needs_translation: bool, detected_language: str)
        """
        lang = self.detect_language(text)
        target = config["target_language"]

        if lang == "unknown":
            return False, lang

        # Don't translate if already in target language
        if lang == target:
            return False, lang

        return True, lang

    def translate(self, text, source_lang="auto", target_lang=None):
        """Translate text to the target language.

        Args:
            text: Text to translate.
            source_lang: Source language code or 'auto' for detection.
            target_lang: Target language code. None uses config default.

        Returns:
            str: Translated text, or original text on failure.
        """
        if not text or not text.strip():
            return text

        target = target_lang or config["target_language"]

        # Check cache
        cache_key = (text.strip(), source_lang, target)
        cached = self._cache.get(cache_key)
        if cached:
            logger.debug("Cache hit for translation")
            return cached

        # Choose translation backend
        mode = config["translation_mode"]
        try:
            if mode == "offline":
                result = self._translate_offline(text, target)
            else:
                result = self._translate_online(text, source_lang, target)
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return text  # return original on failure

        # Cache result
        self._cache.put(cache_key, result)
        return result

    def _translate_online(self, text, source_lang, target_lang):
        """Translate using Google Translate via deep-translator."""
        from deep_translator import GoogleTranslator

        translator = GoogleTranslator(source=source_lang, target=target_lang)

        # deep-translator has a 5000 char limit per request
        if len(text) > 4500:
            chunks = self._split_text(text, 4500)
            results = [translator.translate(chunk) for chunk in chunks]
            return "\n".join(results)

        result = translator.translate(text)
        return result or text

    def _translate_offline(self, text, target_lang):
        """Translate using HuggingFace MarianMT model (offline)."""
        if self._offline_model is None:
            self._load_offline_model(target_lang)

        inputs = self._offline_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        outputs = self._offline_model.generate(**inputs)
        result = self._offline_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result

    def _load_offline_model(self, target_lang):
        """Load HuggingFace MarianMT model for offline translation."""
        try:
            from transformers import MarianMTModel, MarianTokenizer

            model_name = f"Helsinki-NLP/opus-mt-mul-{target_lang}"
            logger.info(f"Loading offline model: {model_name}")
            self._offline_tokenizer = MarianTokenizer.from_pretrained(model_name)
            self._offline_model = MarianMTModel.from_pretrained(model_name)
            logger.info("Offline model loaded successfully")
        except ImportError:
            raise RuntimeError(
                "Offline translation requires: pip install transformers sentencepiece torch"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load offline model: {e}")

    @staticmethod
    def _split_text(text, max_length):
        """Split text into chunks at sentence boundaries."""
        sentences = text.replace("\n", " \n ").split(". ")
        chunks = []
        current = ""
        for sentence in sentences:
            if len(current) + len(sentence) > max_length:
                if current:
                    chunks.append(current.strip())
                current = sentence
            else:
                current += ". " + sentence if current else sentence
        if current:
            chunks.append(current.strip())
        return chunks

    def process(self, text):
        """Full pipeline: detect language → translate if needed.

        Args:
            text: Input text.

        Returns:
            dict: {
                'original': str,
                'translated': str or None,
                'source_lang': str,
                'target_lang': str,
                'was_translated': bool
            }
        """
        needs, source_lang = self.needs_translation(text)

        result = {
            "original": text,
            "translated": None,
            "source_lang": source_lang,
            "target_lang": config["target_language"],
            "was_translated": False,
        }

        if needs:
            translated = self.translate(text, source_lang=source_lang)
            result["translated"] = translated
            result["was_translated"] = True

        return result

    @property
    def cache_size(self):
        return self._cache.size

    def clear_cache(self):
        self._cache.clear()


# Module-level convenience instance
translator = Translator()
