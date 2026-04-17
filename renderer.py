"""
renderer.py — Renders translated text onto screenshot images.

Takes a screenshot image, OCR-detected text regions with bounding boxes,
and their translations, then paints the translations over the original
foreign text — producing a "translated screenshot" image.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import logging

logger = logging.getLogger(__name__)


def _get_font(size=14):
    """Get a good font for rendering text. Falls back gracefully."""
    font_paths = [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _get_dominant_color(image, x, y, w, h):
    """Sample the dominant background color around a text region.

    Looks at the border pixels around the bounding box to estimate
    the chat bubble / background color.
    """
    img_h, img_w = image.shape[:2]

    # Expand region slightly to sample background
    pad = 5
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(img_w, x + w + pad)
    y2 = min(img_h, y + h + pad)

    # Sample border pixels (top row, bottom row, left col, right col)
    samples = []

    # Top border
    if y1 < y:
        samples.extend(image[y1:y, x1:x2].reshape(-1, 3).tolist())
    # Bottom border
    if y + h < y2:
        samples.extend(image[y + h:y2, x1:x2].reshape(-1, 3).tolist())
    # Left border
    if x1 < x:
        samples.extend(image[y1:y2, x1:x].reshape(-1, 3).tolist())
    # Right border
    if x + w < x2:
        samples.extend(image[y1:y2, x + w:x2].reshape(-1, 3).tolist())

    if not samples:
        return (240, 240, 240)  # default light gray

    # Use median for robustness
    samples = np.array(samples)
    median = np.median(samples, axis=0).astype(int)
    return tuple(median.tolist())


def _get_text_color(bg_color):
    """Choose black or white text based on background brightness."""
    brightness = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
    return (255, 255, 255) if brightness < 128 else (0, 0, 0)


def _wrap_text(text, font, max_width, draw):
    """Word-wrap text to fit within max_width pixels."""
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        test = f"{current_line} {word}".strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        tw = bbox[2] - bbox[0]
        if tw <= max_width and current_line:
            current_line = test
        elif not current_line:
            current_line = word
        else:
            lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)
    return lines


def group_words_into_lines(ocr_results, y_threshold=10):
    """Group individual OCR word results into text lines based on Y proximity.

    Args:
        ocr_results: List of dicts with {text, x, y, w, h, conf}.
        y_threshold: Max vertical pixel distance to consider same line.

    Returns:
        List of dicts: {text, x, y, w, h, words: [...]}, one per line.
    """
    if not ocr_results:
        return []

    # Sort by Y then X
    sorted_results = sorted(ocr_results, key=lambda r: (r["y"], r["x"]))

    lines = []
    current_line = {
        "words": [sorted_results[0]],
        "y_center": sorted_results[0]["y"] + sorted_results[0]["h"] // 2,
    }

    for word in sorted_results[1:]:
        word_y_center = word["y"] + word["h"] // 2

        if abs(word_y_center - current_line["y_center"]) <= y_threshold:
            current_line["words"].append(word)
            # Update center as running average
            n = len(current_line["words"])
            current_line["y_center"] = (
                current_line["y_center"] * (n - 1) + word_y_center
            ) / n
        else:
            lines.append(current_line)
            current_line = {"words": [word], "y_center": word_y_center}

    lines.append(current_line)

    # Convert to line-level bounding boxes
    result = []
    for line in lines:
        words = sorted(line["words"], key=lambda w: w["x"])
        text = " ".join(w["text"] for w in words)

        x = min(w["x"] for w in words)
        y = min(w["y"] for w in words)
        x2 = max(w["x"] + w["w"] for w in words)
        y2 = max(w["y"] + w["h"] for w in words)

        avg_conf = sum(w["conf"] for w in words) / len(words)

        result.append({
            "text": text,
            "x": x,
            "y": y,
            "w": x2 - x,
            "h": y2 - y,
            "conf": avg_conf,
            "words": words,
        })

    return result


def group_lines_into_blocks(lines, y_gap_threshold=25, x_overlap_threshold=50):
    """Group nearby lines into text blocks (chat bubbles).

    Args:
        lines: Output from group_words_into_lines().
        y_gap_threshold: Max vertical gap between lines to group.
        x_overlap_threshold: Min horizontal overlap to consider same block.

    Returns:
        List of dicts: {text, x, y, w, h, lines: [...]}, one per block.
    """
    if not lines:
        return []

    # Sort by Y
    sorted_lines = sorted(lines, key=lambda l: l["y"])

    blocks = []
    current_block = [sorted_lines[0]]

    for line in sorted_lines[1:]:
        prev = current_block[-1]

        # Check vertical proximity
        y_gap = line["y"] - (prev["y"] + prev["h"])

        # Check horizontal overlap
        overlap_start = max(line["x"], prev["x"])
        overlap_end = min(line["x"] + line["w"], prev["x"] + prev["w"])
        x_overlap = max(0, overlap_end - overlap_start)

        if y_gap <= y_gap_threshold and x_overlap >= x_overlap_threshold:
            current_block.append(line)
        else:
            blocks.append(current_block)
            current_block = [line]

    blocks.append(current_block)

    # Convert to block objects
    result = []
    for block_lines in blocks:
        text = "\n".join(l["text"] for l in block_lines)
        x = min(l["x"] for l in block_lines)
        y = min(l["y"] for l in block_lines)
        x2 = max(l["x"] + l["w"] for l in block_lines)
        y2 = max(l["y"] + l["h"] for l in block_lines)

        result.append({
            "text": text,
            "x": x,
            "y": y,
            "w": x2 - x,
            "h": y2 - y,
            "lines": block_lines,
        })

    return result


def render_translations(image, blocks, translations):
    """Paint translated text over the original foreign text in the image.

    Args:
        image: numpy.ndarray (BGR format) — the original screenshot.
        blocks: List of text block dicts from group_lines_into_blocks().
        translations: Dict mapping block index → {translated_text, source_lang}.

    Returns:
        numpy.ndarray: Modified image with translations painted on.
    """
    # Convert BGR to RGB for Pillow
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    for idx, block in enumerate(blocks):
        if idx not in translations:
            continue

        trans = translations[idx]
        translated_text = trans["translated_text"]
        source_lang = trans.get("source_lang", "?")

        x, y, w, h = block["x"], block["y"], block["w"], block["h"]

        # Get background color from surrounding pixels
        bg_color = _get_dominant_color(image, x, y, w, h)
        text_color = _get_text_color(bg_color)

        # Choose font size based on block height
        num_lines = max(1, len(block.get("lines", [block])))
        font_size = max(11, min(18, h // num_lines - 4))
        font = _get_font(font_size)

        # Paint background rectangle over original text
        padding = 4
        draw.rectangle(
            [x - padding, y - padding, x + w + padding, y + h + padding],
            fill=tuple(bg_color),
        )

        # Draw thin colored border to indicate translation
        border_color = (0, 150, 255)  # blue accent
        draw.rectangle(
            [x - padding, y - padding, x + w + padding, y + h + padding],
            outline=border_color,
            width=1,
        )

        # Word-wrap the translated text to fit the block width
        wrapped = _wrap_text(translated_text, font, w + padding * 2 - 4, draw)

        # Draw each wrapped line
        line_height = font_size + 3
        ty = y
        for line in wrapped:
            if ty + line_height > y + h + padding * 2:
                # If text overflows, truncate with ellipsis
                draw.text((x, ty), line[:30] + "…", fill=text_color, font=font)
                break
            draw.text((x, ty), line, fill=text_color, font=font)
            ty += line_height

    # Convert back to BGR for OpenCV
    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return result


def render_side_by_side(original, translated, scale=0.5):
    """Create a side-by-side comparison image.

    Args:
        original: Original screenshot (numpy BGR).
        translated: Translated screenshot (numpy BGR).
        scale: Scale factor for the output.

    Returns:
        numpy.ndarray: Side-by-side image.
    """
    h, w = original.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)

    orig_small = cv2.resize(original, (new_w, new_h))
    trans_small = cv2.resize(translated, (new_w, new_h))

    # Add labels
    cv2.putText(orig_small, "Original", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 255), 2)
    cv2.putText(trans_small, "Translated", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)

    combined = np.hstack([orig_small, trans_small])
    return combined
