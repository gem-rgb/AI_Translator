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
import re

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


def _build_line(words):
    """Convert OCR words into a single line object."""
    words = sorted(words, key=lambda w: w["x"])
    text = " ".join(w["text"] for w in words)

    x = min(w["x"] for w in words)
    y = min(w["y"] for w in words)
    x2 = max(w["x"] + w["w"] for w in words)
    y2 = max(w["y"] + w["h"] for w in words)
    avg_conf = sum(w["conf"] for w in words) / len(words)

    return {
        "text": text,
        "x": x,
        "y": y,
        "w": x2 - x,
        "h": y2 - y,
        "conf": avg_conf,
        "words": words,
        "ocr_block_id": words[0].get("ocr_block_id"),
        "ocr_line_id": words[0].get("ocr_line_id"),
    }


def _group_words_by_proximity(sorted_results, y_threshold):
    """Fallback grouping when OCR line metadata is unavailable."""
    lines = []
    current_line = {
        "words": [sorted_results[0]],
        "y_center": sorted_results[0]["y"] + sorted_results[0]["h"] // 2,
    }

    for word in sorted_results[1:]:
        word_y_center = word["y"] + word["h"] // 2

        if abs(word_y_center - current_line["y_center"]) <= y_threshold:
            current_line["words"].append(word)
            n = len(current_line["words"])
            current_line["y_center"] = (
                current_line["y_center"] * (n - 1) + word_y_center
            ) / n
        else:
            lines.append(current_line["words"])
            current_line = {"words": [word], "y_center": word_y_center}

    lines.append(current_line["words"])
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

    grouped_words = []
    line_groups = {}
    for word in sorted_results:
        line_id = word.get("ocr_line_id")
        if not line_id:
            line_groups = {}
            break
        line_groups.setdefault(line_id, []).append(word)

    if line_groups:
        grouped_words = list(line_groups.values())
    else:
        grouped_words = _group_words_by_proximity(sorted_results, y_threshold)

    result = [_build_line(words) for words in grouped_words]
    result.sort(key=lambda line: (line["y"], line["x"]))
    return result


def _horizontal_overlap(a, b):
    """Horizontal overlap between two boxes."""
    overlap_start = max(a["x"], b["x"])
    overlap_end = min(a["x"] + a["w"], b["x"] + b["w"])
    return max(0, overlap_end - overlap_start)


def _looks_sentence_like(text, line_count):
    """Heuristic for message-like or sentence-like content."""
    compact = re.sub(r"\s+", " ", text.strip())
    if not compact:
        return False

    tokens = compact.split()
    letter_count = sum(1 for c in compact if c.isalpha())
    has_sentence_punct = any(c in compact for c in ".!?;,。！？；：…")

    if line_count >= 2:
        return True
    if has_sentence_punct and letter_count >= 4:
        return True
    if len(tokens) >= 4 and letter_count >= 8:
        return True
    if " " not in compact and letter_count >= 6:
        return True
    return False


def _has_consistent_left_edge(lines):
    """Chat bubbles and text paragraphs usually align on the left edge."""
    if len(lines) < 2:
        return False

    left_edges = [line["x"] for line in lines]
    return max(left_edges) - min(left_edges) <= max(18, min(line["h"] for line in lines) * 2)


def _looks_like_ui_label(text, block):
    """Detect short labels, controls, and toolbar fragments."""
    compact = re.sub(r"\s+", " ", text.strip())
    if not compact:
        return True

    token_count = len(compact.split())
    is_single_line = len(block.get("lines", [])) <= 1

    if re.match(r"^\d{1,2}:\d{2}$", compact):
        return True
    if re.match(r"^(https?://|www\.)", compact, re.IGNORECASE):
        return True
    if re.match(r"^[A-Za-z]:\\", compact):
        return True
    if compact.endswith(":") and token_count <= 3:
        return True
    if is_single_line and token_count <= 2 and len(compact) <= 10 and block["w"] < 220:
        return True
    if is_single_line and block["h"] < 28 and block["w"] > 420 and token_count <= 4:
        return True
    return False


def _should_merge_line(current_block, line, y_gap_threshold, x_overlap_threshold):
    """Decide whether a new line belongs in the current text block."""
    prev = current_block[-1]
    same_ocr_block = (
        prev.get("ocr_block_id") is not None
        and prev.get("ocr_block_id") == line.get("ocr_block_id")
    )

    y_gap = line["y"] - (prev["y"] + prev["h"])
    if y_gap > y_gap_threshold * 2:
        return False

    current_bounds = {
        "x": min(item["x"] for item in current_block),
        "y": min(item["y"] for item in current_block),
        "w": max(item["x"] + item["w"] for item in current_block) - min(item["x"] for item in current_block),
        "h": max(item["y"] + item["h"] for item in current_block) - min(item["y"] for item in current_block),
    }
    overlap_with_prev = _horizontal_overlap(prev, line)
    overlap_with_block = _horizontal_overlap(current_bounds, line)
    left_edge_delta = abs(line["x"] - prev["x"])
    center_delta = abs(
        (line["x"] + line["w"] / 2) - (prev["x"] + prev["w"] / 2)
    )

    if same_ocr_block and y_gap <= y_gap_threshold * 2:
        if overlap_with_block >= x_overlap_threshold * 0.4:
            return True
        if left_edge_delta <= max(30, min(prev["h"], line["h"]) * 2):
            return True

    if y_gap <= y_gap_threshold and overlap_with_prev >= x_overlap_threshold:
        return True

    if y_gap <= y_gap_threshold and left_edge_delta <= max(24, min(prev["h"], line["h"]) * 2):
        return True

    if y_gap <= y_gap_threshold and center_delta <= max(prev["w"], line["w"]) * 0.35:
        return True

    return False


def _build_block(block_lines):
    """Convert a list of lines into a block object."""
    text = "\n".join(l["text"] for l in block_lines)
    x = min(l["x"] for l in block_lines)
    y = min(l["y"] for l in block_lines)
    x2 = max(l["x"] + l["w"] for l in block_lines)
    y2 = max(l["y"] + l["h"] for l in block_lines)

    block = {
        "text": text,
        "x": x,
        "y": y,
        "w": x2 - x,
        "h": y2 - y,
        "lines": block_lines,
    }
    block["quality_score"] = _calculate_block_quality(block)
    return block


def group_lines_into_blocks(lines, y_gap_threshold=25, x_overlap_threshold=50):
    """Group nearby lines into text blocks (chat bubbles) with intelligent filtering.

    Args:
        lines: Output from group_words_into_lines().
        y_gap_threshold: Max vertical gap between lines to group.
        x_overlap_threshold: Min horizontal overlap to consider same block.

    Returns:
        List of dicts: {text, x, y, w, h, lines: [...], quality_score}, one per block.
    """
    if not lines:
        return []

    sorted_lines = sorted(lines, key=lambda l: (l["y"], l["x"]))

    blocks = []
    current_block = [sorted_lines[0]]

    for line in sorted_lines[1:]:
        if _should_merge_line(
            current_block,
            line,
            y_gap_threshold=y_gap_threshold,
            x_overlap_threshold=x_overlap_threshold,
        ):
            current_block.append(line)
        else:
            blocks.append(current_block)
            current_block = [line]

    blocks.append(current_block)

    return [_build_block(block_lines) for block_lines in blocks]


def _calculate_block_quality(block):
    """Calculate quality score for a text block to determine translation priority.

    Args:
        block: Block dictionary with text, bounds, and lines.

    Returns:
        float: Quality score (0.0 to 1.0)
    """
    text = block["text"]
    lines = block["lines"]

    if not text.strip():
        return 0.0

    compact = re.sub(r"\s+", " ", text.strip())
    visible_chars = [c for c in compact if not c.isspace()]
    char_count = len(visible_chars)
    if char_count == 0:
        return 0.0

    score = 0.12

    # Factor 1: Average OCR confidence
    avg_confidence = sum(l.get("conf", 50) for l in lines) / len(lines)
    score += min(0.28, max(0.0, avg_confidence / 100) * 0.28)

    # Factor 2: Text density and length
    if char_count >= 50:
        score += 0.18
    elif char_count >= 20:
        score += 0.1
    elif char_count < 5:
        score -= 0.2

    # Factor 3: Word count and line count
    word_count = len(compact.split())
    if word_count >= 5:
        score += 0.14
    elif word_count >= 2:
        score += 0.05

    if len(lines) >= 2:
        score += 0.18

    # Factor 4: Linguistic makeup
    letter_count = sum(1 for c in compact if c.isalpha())
    digit_count = sum(1 for c in compact if c.isdigit())
    symbol_count = sum(1 for c in compact if not c.isalnum() and not c.isspace())
    content_total = max(1, letter_count + digit_count + symbol_count)
    letter_ratio = letter_count / content_total
    digit_ratio = digit_count / max(1, char_count)

    if letter_ratio >= 0.55:
        score += 0.12
    elif letter_ratio < 0.25:
        score -= 0.18

    if digit_ratio > 0.45:
        score -= 0.22

    # Factor 5: Layout hints
    if _has_consistent_left_edge(lines):
        score += 0.08

    if block["w"] >= 140 and block["h"] >= 28:
        score += 0.05

    if _looks_sentence_like(compact, len(lines)):
        score += 0.14

    # Factor 6: Penalize likely UI fragments
    if _looks_like_ui_label(compact, block):
        score -= 0.28

    return max(0.0, min(1.0, score))


def _score_focus_zone(zone_blocks):
    """Score a horizontal focus zone based on quality and message density."""
    score = 0.0
    for block in zone_blocks:
        line_bonus = min(0.2, max(0, len(block.get("lines", [])) - 1) * 0.08)
        length_bonus = min(0.18, len(re.sub(r"\s+", "", block["text"])) / 120)
        score += block["quality_score"] + line_bonus + length_bonus
    return score


def _select_focus_zones(blocks, image_width):
    """Find the horizontal areas most likely to contain real content."""
    if not blocks:
        return []

    candidates = [block for block in blocks if block.get("quality_score", 0) >= 0.25]
    if not candidates:
        candidates = list(blocks)

    padding = max(24, int(image_width * 0.04))
    spans = []
    for block in candidates:
        spans.append({
            "start": max(0, block["x"] - padding),
            "end": min(image_width, block["x"] + block["w"] + padding),
            "blocks": [block],
        })

    spans.sort(key=lambda item: item["start"])
    merged = [spans[0]]
    for span in spans[1:]:
        current = merged[-1]
        if span["start"] <= current["end"]:
            current["end"] = max(current["end"], span["end"])
            current["blocks"].extend(span["blocks"])
        else:
            merged.append(span)

    best_score = max(_score_focus_zone(zone["blocks"]) for zone in merged)
    selected = []
    for zone in merged:
        zone_score = _score_focus_zone(zone["blocks"])
        if zone_score >= best_score * 0.6:
            selected.append(zone)
    return selected


def filter_blocks_by_quality(blocks, image_shape=None, min_quality=0.3, max_blocks=20):
    """Filter text blocks by quality score to reduce noise.

    Args:
        blocks: List of block dictionaries from group_lines_into_blocks().
        image_shape: Optional image shape for context-aware filtering.
        min_quality: Minimum quality score to include.
        max_blocks: Maximum number of blocks to process.

    Returns:
        List of filtered block dictionaries.
    """
    if not blocks:
        return []

    contextual_blocks = [dict(block) for block in blocks]

    if image_shape is not None:
        image_h, image_w = image_shape[:2]
        focus_zones = _select_focus_zones(contextual_blocks, image_w)

        for block in contextual_blocks:
            in_focus_zone = any(
                block["x"] + block["w"] / 2 >= zone["start"]
                and block["x"] + block["w"] / 2 <= zone["end"]
                for zone in focus_zones
            )

            nearby_blocks = [
                other for other in contextual_blocks
                if other is not block
                and abs(other["y"] - block["y"]) <= image_h * 0.18
                and (
                    _horizontal_overlap(block, other) > 0
                    or abs(
                        (other["x"] + other["w"] / 2)
                        - (block["x"] + block["w"] / 2)
                    ) <= image_w * 0.12
                )
            ]

            if in_focus_zone:
                block["quality_score"] = min(1.0, block["quality_score"] + 0.12)
            elif block["quality_score"] < 0.7:
                block["quality_score"] = max(0.0, block["quality_score"] - 0.12)

            if nearby_blocks and any(other["quality_score"] >= 0.5 for other in nearby_blocks):
                block["quality_score"] = min(1.0, block["quality_score"] + 0.06)

            if (
                len(block.get("lines", [])) <= 1
                and len(re.sub(r"\s+", "", block["text"])) <= 8
                and not nearby_blocks
            ):
                block["quality_score"] = max(0.0, block["quality_score"] - 0.1)

    filtered = [block for block in contextual_blocks if block.get("quality_score", 0) >= min_quality]
    filtered.sort(key=lambda b: (b.get("quality_score", 0), b.get("y", 0)), reverse=True)
    return filtered[:max_blocks]


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
