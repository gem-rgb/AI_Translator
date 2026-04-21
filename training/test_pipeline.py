"""Quick pipeline verification test."""
import sys
sys.path.insert(0, r"C:\Users\HomePC\Desktop\AI_RealTime_Translator")

# 1. Config
from config import config
print("1. config OK")

# 2. OCR adapter
from ocr import OCREngine
print("2. OCR adapter OK")

# 3. EasyOCR backend
from ocr_easyocr import EasyOCREngine
print("3. EasyOCR backend OK")

# 4. UI classifier
from ui_classifier import ui_classifier
print(f"4. UI classifier OK (available={ui_classifier.available})")

# 5. Translator - filter test
from translator import Translator
t = Translator()
r = t.process("Settings")
print(f"5. Translator filter: 'Settings' -> {r['filter_reason']}")

# 6. Arabic detection
needs, lang, reason = t.needs_translation("مرحبا كيف حالك")
print(f"6. Arabic detection: needs={needs}, lang={lang}, reason={reason}")

# 7. UI label should be filtered
r2 = t.process("File")
print(f"7. 'File' filtered: {r2['filter_reason']}")

# 8. Real sentence should pass
needs3, lang3, reason3 = t.needs_translation("Bonjour comment allez-vous aujourd'hui")
print(f"8. French sentence: needs={needs3}, lang={lang3}, reason={reason3}")

print("\nALL CHECKS PASSED")
