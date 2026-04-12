"""Narration/segment alignment helpers."""

from __future__ import annotations

import re

from tools.models import ProductionScene, Segment

WORD_RE = re.compile(r"\b[\w']+\b")


def strip_production_tags(text: str) -> str:
    """Remove inline production tags from a scene script."""
    return re.sub(r"\[(?:MODE|VISUAL|TEXT|AVATAR|PACE):[^\]]*\]", "", text)


def normalize_narration(text: str) -> str:
    """Normalize narration text for chunk coverage comparisons."""
    clean = strip_production_tags(text)
    tokens = WORD_RE.findall(clean.lower())
    return " ".join(tokens)


def count_words(text: str) -> int:
    """Count words in narration after tag stripping."""
    return len(WORD_RE.findall(strip_production_tags(text)))


def validate_segment_coverage(scene: ProductionScene, segments: list[Segment]) -> list[str]:
    """Return alignment issues between a scene and its narration chunks."""
    issues: list[str] = []
    if not segments:
        return [f"Scene '{scene.scene}' has no segments"]

    normalized_scene = normalize_narration(scene.script)
    normalized_chunks = " ".join(normalize_narration(seg.narration_chunk) for seg in segments)

    if normalized_scene != normalized_chunks:
        issues.append(
            f"Scene '{scene.scene}' narration chunks do not exactly reconstruct the scene narration"
        )

    for seg in segments:
        if not seg.narration_chunk.strip():
            issues.append(
                f"Scene '{scene.scene}' segment {seg.segment_index} has an empty narration chunk"
            )

    return issues


def assign_segment_timings(scene: ProductionScene, segments: list[Segment]) -> list[Segment]:
    """Derive segment durations and offsets from chunk word counts."""
    if not segments:
        return segments

    total_seconds = max(scene.duration_minutes * 60, 0.1)
    word_counts = [max(1, count_words(seg.narration_chunk)) for seg in segments]
    total_words = sum(word_counts) or len(segments)
    durations = [total_seconds * (words / total_words) for words in word_counts]

    start = 0.0
    for seg, word_count, duration in zip(segments, word_counts, durations):
        seg.word_count = word_count
        seg.duration_seconds = round(duration, 3)
        seg.start_seconds = round(start, 3)
        seg.end_seconds = round(start + duration, 3)
        start += duration

    if segments:
        segments[-1].end_seconds = round(total_seconds, 3)
        segments[-1].duration_seconds = round(
            max(total_seconds - segments[-1].start_seconds, 0.1),
            3,
        )

    return segments
