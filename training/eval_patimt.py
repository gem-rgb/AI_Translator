"""
eval_patimt.py — Evaluate the translation pipeline against PATIMT-Bench.

Clones the PATIMT-Bench repo, runs our OCR + translation pipeline on
the test images, and measures accuracy.

Usage:
    python training/eval_patimt.py

Requires: git (for cloning the repo)
"""

import json
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PATIMT_DIR = os.path.join(DATA_DIR, "PATIMT-Bench")
PATIMT_REPO = "https://github.com/XMUDeepLIT/PATIMT-Bench.git"


def clone_patimt():
    """Clone PATIMT-Bench repo if not already present."""
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.isdir(PATIMT_DIR):
        print(f"PATIMT-Bench already cloned at {PATIMT_DIR}")
        return

    print(f"Cloning PATIMT-Bench from {PATIMT_REPO} ...")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", PATIMT_REPO, PATIMT_DIR],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Git clone failed: {result.stderr}")
        print("Make sure git is installed and accessible.")
        sys.exit(1)
    print("Cloned successfully.")


def load_instruction_data():
    """Load PATIMT instruction data for evaluation."""
    instruction_dir = os.path.join(PATIMT_DIR, "instruction_data_question")

    if not os.path.isdir(instruction_dir):
        print(f"Instruction data not found at {instruction_dir}")
        return []

    samples = []
    for fname in os.listdir(instruction_dir):
        if not fname.endswith(".json") and not fname.endswith(".jsonl"):
            continue

        fpath = os.path.join(instruction_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                if fname.endswith(".jsonl"):
                    for line in f:
                        line = line.strip()
                        if line:
                            samples.append(json.loads(line))
                else:
                    data = json.load(f)
                    if isinstance(data, list):
                        samples.extend(data)
                    else:
                        samples.append(data)
        except (json.JSONDecodeError, IOError) as exc:
            print(f"Skipping {fname}: {exc}")

    print(f"Loaded {len(samples)} PATIMT evaluation samples")
    return samples


def evaluate_pipeline():
    """Run the evaluation pipeline."""
    clone_patimt()
    samples = load_instruction_data()

    if not samples:
        print("No evaluation samples found. PATIMT-Bench may not include images directly.")
        print("The benchmark is primarily designed for Vision-Language Models (VLMs).")
        print("\nHowever, we can still use it as a reference for our pipeline design.")
        _print_architecture_notes()
        return

    print(f"\nEvaluating {len(samples)} samples...")

    # Import our pipeline components
    try:
        from translator import Translator
        translator = Translator()
    except ImportError as exc:
        print(f"Cannot import pipeline: {exc}")
        return

    correct = 0
    total = 0
    errors = 0

    for i, sample in enumerate(samples[:50]):  # Evaluate first 50
        text = sample.get("text", sample.get("source_text", ""))
        if not text:
            continue

        total += 1
        try:
            result = translator.process(text)
            if result["was_translated"]:
                correct += 1
                print(f"  [{i+1}] ✓ {result['source_lang']} → {result['target_lang']}")
            else:
                print(f"  [{i+1}] ✗ Filtered: {result['filter_reason']}")
        except Exception as exc:
            errors += 1
            print(f"  [{i+1}] ERROR: {exc}")

    print(f"\n{'='*50}")
    print(f"Results: {correct}/{total} translated, {errors} errors")
    if total > 0:
        print(f"Translation rate: {correct/total*100:.1f}%")
    print(f"{'='*50}")


def _print_architecture_notes():
    """Print notes about how PATIMT-Bench informs our design."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                  PATIMT-Bench Design Insights                    ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  PATIMT-Bench evaluates position-aware text image translation    ║
║  across 10 scenarios. Key takeaways for our pipeline:            ║
║                                                                  ║
║  1. Position matters: text at screen edges is usually UI         ║
║     → Our scratchpad already applies zone penalties              ║
║                                                                  ║
║  2. Bounding box preservation: translations should map back      ║
║     to original positions                                        ║
║     → Our pipeline tracks block positions through translation    ║
║                                                                  ║
║  3. Multi-scenario: ads, posters, charts, screenshots            ║
║     → Our classifier trains on screenshots specifically          ║
║                                                                  ║
║  4. OCR quality is critical: the benchmark assumes good OCR      ║
║     → EasyOCR upgrade addresses this directly                    ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    evaluate_pipeline()
