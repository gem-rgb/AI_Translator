"""
scratchpad.py - Hidden frame-to-frame decision memory for screenshot analysis.

This module keeps lightweight structured observations from recent screenshots so
the app can make better decisions about which OCR blocks are likely real chat
content worth translating. It is designed to be compatible with a future
learned model, but currently uses transparent heuristics.
"""

from collections import deque
import hashlib
import re

import cv2

from config import config


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


class ScreenScratchpad:
    """Keeps recent screenshot-derived observations for smarter selection."""

    def __init__(self):
        self._history = deque(maxlen=int(config.get("scratchpad_history", 8)))
        self._tracks = {}
        self._frame_index = 0
        self._last_frame_hash = None

        self._script_patterns = {
            "arabic": re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]"),
            "cyrillic": re.compile(r"[\u0400-\u04FF]"),
            "cjk": re.compile(r"[\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uAC00-\uD7AF]"),
            "devanagari": re.compile(r"[\u0900-\u097F]"),
        }

    def observe(self, image, blocks):
        """Analyze the current frame and return selected translation candidates."""
        frame_hash = self._hash_frame(image)
        if frame_hash == self._last_frame_hash and self._history:
            previous = self._history[-1]
            return {
                "selected_blocks": [dict(block) for block in previous["selected_blocks"]],
                "summary": previous["summary"],
                "focus_lanes": list(previous["focus_lanes"]),
                "reused": True,
            }

        self._frame_index += 1
        image_h, image_w = image.shape[:2]
        analyzed_blocks = [self._prepare_block(block, image_w, image_h) for block in blocks]

        focus_lanes = self._derive_focus_lanes(analyzed_blocks, image_w)
        self._update_tracks(analyzed_blocks)
        selected_blocks = self._select_blocks(analyzed_blocks, focus_lanes, image_w, image_h)
        self._prune_tracks()

        summary = (
            f"scratchpad: {len(selected_blocks)} candidates, "
            f"{len(focus_lanes)} focus lane(s), {len(self._tracks)} active track(s)"
        )

        record = {
            "frame_hash": frame_hash,
            "selected_blocks": [dict(block) for block in selected_blocks],
            "focus_lanes": list(focus_lanes),
            "summary": summary,
        }
        self._history.append(record)
        self._last_frame_hash = frame_hash

        return {
            "selected_blocks": selected_blocks,
            "summary": summary,
            "focus_lanes": focus_lanes,
            "reused": False,
        }

    def reset(self):
        """Clear accumulated memory."""
        self._history.clear()
        self._tracks.clear()
        self._frame_index = 0
        self._last_frame_hash = None

    def _prepare_block(self, block, image_w, image_h):
        """Attach normalized analysis features to a block."""
        text = block.get("text", "").strip()
        compact = re.sub(r"\s+", " ", text)
        fingerprint = self._fingerprint_text(compact)
        script_family = self._detect_script_family(compact)
        line_count = len(block.get("lines", [])) or 1
        char_count = len(re.sub(r"\s+", "", compact))

        prepared = dict(block)
        prepared.update({
            "compact_text": compact,
            "fingerprint": fingerprint,
            "script_family": script_family,
            "line_count": line_count,
            "char_count": char_count,
            "x_norm": prepared["x"] / max(1, image_w),
            "y_norm": prepared["y"] / max(1, image_h),
            "w_norm": prepared["w"] / max(1, image_w),
            "h_norm": prepared["h"] / max(1, image_h),
            "center_x": prepared["x"] + prepared["w"] / 2,
            "center_y": prepared["y"] + prepared["h"] / 2,
            "track_seen_count": 0,
            "track_recent_hits": 0,
            "scratchpad_score": prepared.get("quality_score", 0.0),
        })
        return prepared

    def _derive_focus_lanes(self, blocks, image_w):
        """Estimate horizontal lanes where meaningful chat text lives."""
        target_script = TARGET_SCRIPT_FAMILY.get(config["target_language"], "latin")
        candidates = []

        for frame in self._history:
            for block in frame.get("selected_blocks", []):
                weight = block.get("scratchpad_score", block.get("quality_score", 0.0))
                if block.get("script_family") not in ("latin", target_script):
                    weight += 0.18
                candidates.append({
                    "x": block["x"],
                    "w": block["w"],
                    "weight": weight,
                })

        for block in blocks:
            if block.get("quality_score", 0.0) >= 0.3:
                weight = block.get("quality_score", 0.0)
                if block["script_family"] not in ("latin", target_script):
                    weight += 0.25
                if block["y_norm"] < 0.14 and block["w_norm"] > 0.55:
                    weight *= 0.45
                if block["x_norm"] < 0.15 and block["w_norm"] < 0.22:
                    weight *= 0.65
                candidates.append({
                    "x": block["x"],
                    "w": block["w"],
                    "weight": weight,
                })

        if not candidates:
            return []

        padding = max(24, int(image_w * 0.04))
        spans = []
        for candidate in candidates:
            spans.append({
                "start": max(0, candidate["x"] - padding),
                "end": min(image_w, candidate["x"] + candidate["w"] + padding),
                "weight": candidate["weight"],
            })

        spans.sort(key=lambda item: item["start"])
        merged = [spans[0]]
        for span in spans[1:]:
            current = merged[-1]
            if span["start"] <= current["end"]:
                current["end"] = max(current["end"], span["end"])
                current["weight"] += span["weight"]
            else:
                merged.append(span)

        best_weight = max(item["weight"] for item in merged)
        selected = []
        for lane in merged:
            if lane["weight"] >= best_weight * 0.55:
                selected.append((lane["start"], lane["end"]))
        return selected[:2]

    def _update_tracks(self, blocks):
        """Track repeated text candidates over time."""
        for block in blocks:
            fingerprint = block.get("fingerprint")
            if not fingerprint:
                continue

            track = self._tracks.get(fingerprint)
            if track is None:
                self._tracks[fingerprint] = {
                    "fingerprint": fingerprint,
                    "seen_count": 1,
                    "recent_hits": 1,
                    "last_seen_frame": self._frame_index,
                    "avg_x": block["x"],
                    "avg_w": block["w"],
                    "script_family": block["script_family"],
                    "quality_score": block.get("quality_score", 0.0),
                }
                block["track_seen_count"] = 1
                block["track_recent_hits"] = 1
                continue

            gap = self._frame_index - track["last_seen_frame"]
            if gap <= 2:
                track["recent_hits"] = min(track["recent_hits"] + 1, 6)
            else:
                track["recent_hits"] = 1

            track["seen_count"] += 1
            track["last_seen_frame"] = self._frame_index
            track["avg_x"] = (track["avg_x"] * 0.7) + (block["x"] * 0.3)
            track["avg_w"] = (track["avg_w"] * 0.7) + (block["w"] * 0.3)
            track["quality_score"] = max(track["quality_score"], block.get("quality_score", 0.0))

            block["track_seen_count"] = track["seen_count"]
            block["track_recent_hits"] = track["recent_hits"]

    @staticmethod
    def _neighbor_count(blocks, target, image_h):
        """Count nearby vertically separated blocks in a similar lane."""
        count = 0
        for other in blocks:
            if other is target:
                continue

            same_lane = abs(other["center_x"] - target["center_x"]) <= max(target["w"], other["w"]) * 0.7
            vertical_gap = abs(other["center_y"] - target["center_y"])

            if same_lane and 18 <= vertical_gap <= image_h * 0.35:
                count += 1
        return count

    def _select_blocks(self, blocks, focus_lanes, image_w, image_h):
        """Choose the blocks that look most like real foreign-language content."""
        target_script = TARGET_SCRIPT_FAMILY.get(config["target_language"], "latin")
        selected = []
        min_score = float(config.get("scratchpad_min_score", 0.48))

        for block in blocks:
            score = block.get("quality_score", 0.0)
            neighbor_count = self._neighbor_count(blocks, block, image_h)
            in_lane = any(
                block["center_x"] >= lane_start and block["center_x"] <= lane_end
                for lane_start, lane_end in focus_lanes
            )

            if in_lane:
                score += 0.12
            elif score < 0.75:
                score -= 0.08

            if block["track_seen_count"] >= 2:
                score += 0.1
            if block["track_recent_hits"] >= 2:
                score += 0.08

            if neighbor_count >= 2:
                score += 0.12
            elif neighbor_count == 1:
                score += 0.05
            elif block["line_count"] == 1:
                score -= 0.08

            if block["line_count"] >= 2:
                score += 0.07
            elif block["char_count"] <= 5:
                score -= 0.08

            if block["script_family"] != "latin" and block["script_family"] != target_script:
                score += 0.22

            if block["w"] >= max(160, int(image_w * 0.14)):
                score += 0.04

            if block["y_norm"] < 0.08:
                score -= 0.04

            if block["y_norm"] < 0.12 and block["w_norm"] > 0.45 and block["h_norm"] < 0.07:
                score -= 0.42
            elif block["y_norm"] < 0.16 and block["w_norm"] > 0.55 and block["h_norm"] < 0.09:
                score -= 0.28

            if block["w_norm"] > 0.78 and block["line_count"] <= 1:
                score -= 0.16

            if block["x_norm"] < 0.16 and block["w_norm"] < 0.22:
                score -= 0.1

            block["scratchpad_score"] = max(0.0, min(1.0, score))

            threshold = min_score
            if block["script_family"] != "latin" and block["script_family"] != target_script:
                threshold -= 0.07

            if block["scratchpad_score"] >= threshold:
                selected.append(block)

        selected.sort(key=lambda item: (item["y"], item["x"]))
        return selected[:12]

    def _prune_tracks(self):
        """Remove stale tracks that are no longer relevant."""
        ttl = int(config.get("scratchpad_track_ttl", 6))
        to_delete = []
        for fingerprint, track in self._tracks.items():
            if self._frame_index - track["last_seen_frame"] > ttl:
                to_delete.append(fingerprint)

        for fingerprint in to_delete:
            del self._tracks[fingerprint]

    @staticmethod
    def _fingerprint_text(text):
        """Create a stable fingerprint for OCR text across nearby frames."""
        normalized = re.sub(r"[^\w\u0600-\u06FF\u0400-\u04FF\u3040-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF]+", "", text.lower())
        return normalized[:80]

    def _detect_script_family(self, text):
        """Detect a broad script family from unicode ranges."""
        for family, pattern in self._script_patterns.items():
            if pattern.search(text):
                return family
        return "latin"

    @staticmethod
    def _hash_frame(image):
        """Small perceptual hash for frame reuse checks."""
        small = cv2.resize(image, (24, 24))
        return hashlib.md5(small.tobytes()).hexdigest()
