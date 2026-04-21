"""
download_rico.py — Download and preprocess the Rico dataset for UI classifier training.

Downloads the Enrico subset (curated 1,460 UIs from Rico) which is smaller and
faster to download than the full 66K Rico dataset, but sufficient for training
a high-quality UI-element vs content classifier.

Usage:
    python training/download_rico.py

Output:
    training/data/rico_features.csv
"""

import csv
import json
import os
import sys
import urllib.request
import zipfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
ENRICO_URL = "https://github.com/luileito/enrico/archive/refs/heads/master.zip"
ENRICO_ZIP = os.path.join(DATA_DIR, "enrico.zip")
ENRICO_DIR = os.path.join(DATA_DIR, "enrico-master")
OUTPUT_CSV = os.path.join(DATA_DIR, "rico_features.csv")

# UI element types from Rico view hierarchy that are clearly NOT chat content
UI_ELEMENT_CLASSES = {
    "Toolbar", "Image", "Icon", "Background Image", "Advertisement",
    "Drawer", "Bottom Navigation", "Tab Bar", "Modal", "Pager Indicator",
    "On/Off Switch", "Slider", "Progress Bar", "Rating", "Number Stepper",
    "Map View", "Web View", "Video", "Date Picker",
}

# Text-bearing UI elements — labels, buttons, etc.
UI_TEXT_CLASSES = {
    "Text Button", "Icon Button", "Button Bar",
    "Toolbar Title", "Page Title",
}

# Content that could be real user-generated text
CONTENT_CLASSES = {
    "Text", "Text Area", "Input", "Card",
    "List Item", "Multi-Tab", "Section Header",
}


def download_enrico():
    """Download the Enrico dataset if not already present."""
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.isdir(ENRICO_DIR):
        print(f"Enrico already downloaded at {ENRICO_DIR}")
        return

    print(f"Downloading Enrico dataset from {ENRICO_URL} ...")
    urllib.request.urlretrieve(ENRICO_URL, ENRICO_ZIP)
    print(f"Downloaded to {ENRICO_ZIP}")

    print("Extracting...")
    with zipfile.ZipFile(ENRICO_ZIP, "r") as zf:
        zf.extractall(DATA_DIR)
    print(f"Extracted to {ENRICO_DIR}")

    # Clean up zip
    os.remove(ENRICO_ZIP)


def parse_hierarchy(hierarchy_path, screen_w=1440, screen_h=2560):
    """Parse a Rico/Enrico view hierarchy JSON and extract training samples.

    Each sample is a text-bearing UI element with features:
      - x_norm, y_norm, w_norm, h_norm (position/size normalised to screen)
      - text_length, word_count
      - element_class (from Rico)
      - label: 1 = UI element, 0 = content
    """
    try:
        with open(hierarchy_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return []

    samples = []
    _walk_node(data, samples, screen_w, screen_h)
    return samples


def _walk_node(node, samples, sw, sh):
    """Recursively walk the view hierarchy tree."""
    if not isinstance(node, dict):
        return

    bounds = node.get("bounds", [0, 0, 0, 0])
    if len(bounds) == 4:
        x1, y1, x2, y2 = bounds
    else:
        x1, y1, x2, y2 = 0, 0, 0, 0

    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    component_label = node.get("componentLabel", "")
    text_content = (node.get("text", "") or "").strip()

    # Only interested in elements that have text or a known component label
    if text_content or component_label:
        # Determine label
        if component_label in UI_ELEMENT_CLASSES or component_label in UI_TEXT_CLASSES:
            label = 1  # UI element
        elif component_label in CONTENT_CLASSES:
            label = 0  # Content
        elif text_content and len(text_content.split()) >= 4:
            label = 0  # Longer text is likely content
        elif text_content and len(text_content) <= 15:
            label = 1  # Short text is likely a label
        else:
            label = -1  # Ambiguous, skip

        if label >= 0 and w > 0 and h > 0:
            samples.append({
                "x_norm": round(x1 / max(1, sw), 4),
                "y_norm": round(y1 / max(1, sh), 4),
                "w_norm": round(w / max(1, sw), 4),
                "h_norm": round(h / max(1, sh), 4),
                "text_length": len(text_content),
                "word_count": len(text_content.split()) if text_content else 0,
                "is_single_line": 1 if "\n" not in text_content else 0,
                "has_punctuation": 1 if any(c in text_content for c in ".!?,;:") else 0,
                "letter_ratio": round(
                    sum(1 for c in text_content if c.isalpha()) / max(1, len(text_content)), 4
                ) if text_content else 0.0,
                "component": component_label,
                "label": label,
            })

    # Recurse into children
    for child in node.get("children", []):
        _walk_node(child, samples, sw, sh)


def generate_features():
    """Process all Enrico hierarchies and output a feature CSV."""
    hierarchy_dir = os.path.join(ENRICO_DIR, "design_topics")

    if not os.path.isdir(hierarchy_dir):
        # Try alternative structure
        for root, dirs, files in os.walk(ENRICO_DIR):
            if any(f.endswith(".json") for f in files):
                hierarchy_dir = root
                break

    all_samples = []
    json_count = 0

    for root, dirs, files in os.walk(ENRICO_DIR):
        for fname in files:
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(root, fname)
            samples = parse_hierarchy(fpath)
            all_samples.extend(samples)
            json_count += 1

    if len(all_samples) < 500:
        real_count = len(all_samples)
        print(f"Only {real_count} real samples — augmenting with synthetic training data...")
        synthetic = _generate_synthetic_data()
        all_samples.extend(synthetic)
        print(f"Added {len(synthetic)} synthetic samples (total: {len(all_samples)})")

    print(f"Processed {json_count} hierarchy files, total {len(all_samples)} samples")

    # Write CSV
    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = [
        "x_norm", "y_norm", "w_norm", "h_norm",
        "text_length", "word_count", "is_single_line",
        "has_punctuation", "letter_ratio", "label",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample in all_samples:
            row = {k: sample[k] for k in fieldnames}
            writer.writerow(row)

    print(f"Saved features to {OUTPUT_CSV}")
    print(f"  UI elements (label=1): {sum(1 for s in all_samples if s['label'] == 1)}")
    print(f"  Content (label=0):     {sum(1 for s in all_samples if s['label'] == 0)}")


def _generate_synthetic_data():
    """Generate synthetic training data if Rico download fails.

    This creates realistic feature distributions based on known UI patterns
    so the classifier can still be trained without the dataset.
    """
    import random
    random.seed(42)
    samples = []

    # --- UI elements: buttons, labels, menu items, toolbar text ---
    for _ in range(3000):
        # Toolbar / title bar region (top of screen)
        y = random.uniform(0.0, 0.12)
        x = random.uniform(0.0, 0.8)
        w = random.uniform(0.05, 0.5)
        h = random.uniform(0.02, 0.06)
        samples.append({
            "x_norm": round(x, 4), "y_norm": round(y, 4),
            "w_norm": round(w, 4), "h_norm": round(h, 4),
            "text_length": random.randint(2, 12),
            "word_count": random.randint(1, 2),
            "is_single_line": 1,
            "has_punctuation": 0,
            "letter_ratio": round(random.uniform(0.8, 1.0), 4),
            "label": 1,
        })

    for _ in range(2000):
        # Bottom navigation / status bar
        y = random.uniform(0.90, 0.98)
        x = random.uniform(0.0, 0.8)
        w = random.uniform(0.08, 0.25)
        h = random.uniform(0.02, 0.05)
        samples.append({
            "x_norm": round(x, 4), "y_norm": round(y, 4),
            "w_norm": round(w, 4), "h_norm": round(h, 4),
            "text_length": random.randint(3, 10),
            "word_count": random.randint(1, 2),
            "is_single_line": 1,
            "has_punctuation": 0,
            "letter_ratio": round(random.uniform(0.7, 1.0), 4),
            "label": 1,
        })

    for _ in range(2000):
        # Sidebar / menu labels
        x = random.uniform(0.0, 0.15)
        y = random.uniform(0.1, 0.9)
        w = random.uniform(0.05, 0.20)
        h = random.uniform(0.02, 0.04)
        samples.append({
            "x_norm": round(x, 4), "y_norm": round(y, 4),
            "w_norm": round(w, 4), "h_norm": round(h, 4),
            "text_length": random.randint(3, 15),
            "word_count": random.randint(1, 3),
            "is_single_line": 1,
            "has_punctuation": 0,
            "letter_ratio": round(random.uniform(0.75, 1.0), 4),
            "label": 1,
        })

    for _ in range(1500):
        # Small scattered buttons
        x = random.uniform(0.0, 0.9)
        y = random.uniform(0.0, 0.95)
        w = random.uniform(0.05, 0.18)
        h = random.uniform(0.02, 0.05)
        samples.append({
            "x_norm": round(x, 4), "y_norm": round(y, 4),
            "w_norm": round(w, 4), "h_norm": round(h, 4),
            "text_length": random.randint(2, 8),
            "word_count": 1,
            "is_single_line": 1,
            "has_punctuation": 0,
            "letter_ratio": round(random.uniform(0.8, 1.0), 4),
            "label": 1,
        })

    # --- Content: chat messages, paragraphs, user text ---
    for _ in range(4000):
        # Chat messages — mid-screen, moderate width, multi-word
        x = random.uniform(0.05, 0.55)
        y = random.uniform(0.15, 0.85)
        w = random.uniform(0.25, 0.65)
        h = random.uniform(0.03, 0.12)
        wc = random.randint(3, 25)
        tl = wc * random.randint(3, 7)
        samples.append({
            "x_norm": round(x, 4), "y_norm": round(y, 4),
            "w_norm": round(w, 4), "h_norm": round(h, 4),
            "text_length": tl,
            "word_count": wc,
            "is_single_line": 1 if wc < 8 else 0,
            "has_punctuation": 1 if random.random() > 0.3 else 0,
            "letter_ratio": round(random.uniform(0.65, 0.95), 4),
            "label": 0,
        })

    for _ in range(2500):
        # Paragraphs — wider, taller, many words
        x = random.uniform(0.03, 0.2)
        y = random.uniform(0.15, 0.75)
        w = random.uniform(0.5, 0.9)
        h = random.uniform(0.06, 0.25)
        wc = random.randint(10, 60)
        tl = wc * random.randint(4, 6)
        samples.append({
            "x_norm": round(x, 4), "y_norm": round(y, 4),
            "w_norm": round(w, 4), "h_norm": round(h, 4),
            "text_length": tl,
            "word_count": wc,
            "is_single_line": 0,
            "has_punctuation": 1,
            "letter_ratio": round(random.uniform(0.7, 0.92), 4),
            "label": 0,
        })

    random.shuffle(samples)
    return samples


if __name__ == "__main__":
    download_enrico()
    generate_features()
