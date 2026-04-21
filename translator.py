"""
translator.py — Language detection and translation engine.

Supports:
  - Online mode: Google Translate via deep-translator (free, no API key)
  - Offline mode: HuggingFace MarianMT models (requires model download)
  - LRU translation cache to avoid redundant API calls
"""

import logging
import re

from langdetect import detect, LangDetectException

from config import config

logger = logging.getLogger(__name__)

TARGET_SCRIPT_FAMILY = {
    "en": "latin",
    "fr": "latin",
    "de": "latin",
    "es": "latin",
    "pt": "latin",
    "tr": "latin",
    "ar": "arabic",
    "ru": "cyrillic",
    "ja": "cjk",
    "zh-CN": "cjk",
    "zh-TW": "cjk",
    "ko": "cjk",
    "hi": "devanagari",
}

# Comprehensive list of common English UI words/labels that should never be
# treated as foreign text requiring translation.  Kept lowercase for fast
# membership testing.
_UI_WORD_BLACKLIST = frozenset({
    # -- Window chrome & system --
    "file", "edit", "view", "insert", "format", "tools", "window", "help",
    "home", "settings", "options", "preferences", "about", "properties",
    "close", "exit", "quit", "minimize", "maximize", "restore",
    # -- Common buttons / actions --
    "ok", "cancel", "yes", "no", "apply", "save", "open", "new", "delete",
    "remove", "add", "create", "update", "submit", "send", "reply", "forward",
    "back", "next", "previous", "finish", "done", "retry", "refresh", "reload",
    "undo", "redo", "cut", "copy", "paste", "select", "search", "find",
    "replace", "print", "share", "export", "import", "download", "upload",
    # -- Navigation / tabs --
    "dashboard", "profile", "account", "inbox", "notifications", "messages",
    "contacts", "favorites", "bookmarks", "history", "recent", "popular",
    "trending", "explore", "discover", "browse", "library", "archive",
    # -- Status / labels --
    "online", "offline", "busy", "away", "active", "inactive", "enabled",
    "disabled", "on", "off", "loading", "processing", "connecting",
    "connected", "disconnected", "error", "warning", "info", "success",
    "failed", "pending", "complete", "completed", "progress", "status",
    "type", "name", "date", "time", "size", "location", "description",
    # -- Media / content --
    "play", "pause", "stop", "mute", "unmute", "volume", "fullscreen",
    "picture", "video", "audio", "image", "photo", "camera", "microphone",
    # -- System tray / taskbar --
    "start", "taskbar", "desktop", "recycle", "bin", "network", "wifi",
    "bluetooth", "battery", "brightness", "sound", "display",
    # -- Browser --
    "tabs", "extensions", "incognito", "private", "bookmark", "address",
    "toolbar", "menu", "more", "zoom", "page", "source", "console",
    "inspect", "developer",
    # -- Chat app UI (not chat content) --
    "chats", "groups", "channels", "calls", "stories", "status",
    "typing", "delivered", "read", "seen", "pinned", "archived", "starred",
    "muted", "blocked", "report", "unread", "draft",
})


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

        # Patterns for text filtering
        self._ui_patterns = [
            r'^[0-9]+$',  # Pure numbers
            r'^[a-zA-Z]$',  # Single letters
            r'^[^\w\s]+$',  # Pure symbols
            r'^\d{1,2}:\d{2}$',  # Time format
            r'^\d+%$',  # Percentage
            r'^(https?://|www\.)',  # URLs
            r'^[A-Za-z]:\\',  # Windows paths
            r'^[@#][\w.-]+$',  # Handles / tags
            r'^[<>]=?\d*$',  # Comparison operators
            r'^[+\-*/=]$',  # Math operators
            r'^(OK|Cancel|Yes|No|Close|Exit|Save|Delete|Edit|View|Help|Settings)$',  # Common UI buttons
        ]

        self._compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self._ui_patterns]
        self._script_patterns = {
            "arabic": re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]"),
            "cyrillic": re.compile(r"[\u0400-\u04FF]"),
            "cjk": re.compile(r"[\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uAC00-\uD7AF]"),
            "devanagari": re.compile(r"[\u0900-\u097F]"),
        }

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

    def _is_meaningful_text(self, text):
        """Check if text is meaningful content worth translating.

        Filters out UI elements, numbers, symbols, and other non-content text.

        Args:
            text: Input text string.

        Returns:
            tuple: (is_meaningful: bool, reason: str)
        """
        if not text or len(text.strip()) < 2:
            return False, "too_short"

        text = re.sub(r"\s+", " ", text.strip())

        # Check against UI patterns
        for pattern in self._compiled_patterns:
            if pattern.match(text):
                return False, "ui_element"

        if text.endswith(":") and len(text.split()) <= 3:
            return False, "label_text"

        # Filter text with very low word content
        words = text.split()
        if len(words) == 1 and len(words[0]) <= 2:
            return False, "single_short_word"

        if len(words) == 1 and len(words[0]) <= 4 and words[0].isascii() and words[0].isupper():
            return False, "short_uppercase_label"

        # --- Expanded UI word blacklist filter ---
        # If the entire text (1-3 words) consists of common UI vocabulary,
        # it is almost certainly a button, menu item, or label — not chat.
        if len(words) <= 3:
            lower_words = [w.lower().rstrip(".:;,") for w in words]
            if all(w in _UI_WORD_BLACKLIST for w in lower_words if w):
                return False, "ui_label_blacklist"

        # Single-word text that is a known UI label
        if len(words) == 1:
            if words[0].lower().rstrip(".:;,") in _UI_WORD_BLACKLIST:
                return False, "ui_single_word"

        # Filter text that's mostly numbers or symbols
        letter_chars = sum(1 for c in text if c.isalpha())
        alnum_chars = sum(1 for c in text if c.isalnum())
        digit_chars = sum(1 for c in text if c.isdigit())
        visible_chars = sum(1 for c in text if not c.isspace())

        if visible_chars > 3 and alnum_chars / visible_chars < 0.3:
            return False, "mostly_symbols"

        if digit_chars / max(1, visible_chars) > 0.45 and letter_chars < 3:
            return False, "mostly_numeric"

        # Filter common version numbers, dates, file extensions
        if re.match(r'^v?\d+\.\d+(\.\d+)?$', text) or re.match(r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$', text):
            return False, "version_or_date"

        if re.match(r'^\S+@\S+\.\S+$', text):
            return False, "email_address"

        # Filter file extensions
        if text.startswith('.') and len(text) <= 5:
            return False, "file_extension"

        if len(words) <= 2 and len(text) <= 10 and text.istitle():
            return False, "short_title_label"

        return True, "meaningful"

    def _detect_script_hint(self, text):
        """Detect a broad writing-system family from unicode ranges."""
        for family, pattern in self._script_patterns.items():
            if pattern.search(text):
                return family
        return "latin"

    def needs_translation(self, text):
        """Check if text needs translation (i.e., is not in the target language).

        Args:
            text: Input text.

        Returns:
            tuple: (needs_translation: bool, detected_language: str, reason: str)
        """
        # First check if text is meaningful
        is_meaningful, reason = self._is_meaningful_text(text)
        if not is_meaningful:
            return False, "filtered", reason

        target = config["target_language"]
        source_script = self._detect_script_hint(text)
        target_script = TARGET_SCRIPT_FAMILY.get(target, "latin")

        # Script-aware fast path for short foreign chat text. This helps when
        # langdetect struggles on compact Arabic/Persian chat messages.
        if source_script != "latin" and source_script != target_script:
            return True, "auto", f"{source_script}_script"

        # --- Short Latin-script guard ---
        # For Latin-script text with fewer than 3 words, langdetect is very
        # unreliable and almost always produces false positives on English UI
        # text.  Skip translation for these — real chat messages are usually
        # longer.
        words = re.sub(r"\s+", " ", text.strip()).split()
        if source_script == "latin" and len(words) < 3:
            return False, "en", "short_latin_skip"

        lang = self.detect_language(text)

        if lang == "unknown":
            return False, lang, "unknown_language"

        # Don't translate if already in target language
        if lang == target:
            return False, lang, "already_target_language"

        return True, lang, "needs_translation"

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
                'was_translated': bool,
                'filter_reason': str or None
            }
        """
        needs, source_lang, reason = self.needs_translation(text)

        result = {
            "original": text,
            "translated": None,
            "source_lang": source_lang,
            "target_lang": config["target_language"],
            "was_translated": False,
            "filter_reason": reason if not needs else None,
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
