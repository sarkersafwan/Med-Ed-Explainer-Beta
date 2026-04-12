"""Quality validation for generated scripts.

Runs deterministic checks — no LLM calls. Catches issues that slip through
the generation process.
"""

from __future__ import annotations

import re

from tools.models import ProductionScript

# Phrases that scream "AI-generated educational content"
SLOP_PHRASES = [
    "let's dive in",
    "in this video",
    "without further ado",
    "it's important to note",
    "as we all know",
    "let's get started",
    "before we begin",
    "in conclusion",
    "to wrap up",
    "let's summarize",
    "to summarize",
    "in today's video",
    "welcome back",
    "hey everyone",
    "buckle up",
    "fasten your seatbelts",
    "shall we",
    "are you ready",
    "let me break it down",
    "without any delay",
]

WPM_MIN = 130
WPM_MAX = 170
MAX_SCENE_MINUTES = 5.0
ALLOWED_MODES = {"avatar", "animation", "overlay"}


def validate_script(script: ProductionScript) -> list[str]:
    """Validate a production script and return a list of issues found.

    Returns empty list if the script passes all checks.
    """
    issues: list[str] = []

    if not script.scenes:
        issues.append("No scenes in script")
        return issues

    for scene in script.scenes:
        prefix = f"Scene '{scene.scene}'"

        # Word count vs duration check
        if scene.duration_minutes > 0:
            expected_wpm = scene.word_count / scene.duration_minutes
            if expected_wpm < WPM_MIN:
                issues.append(
                    f"{prefix}: Word count too low for duration "
                    f"({scene.word_count} words / {scene.duration_minutes} min = "
                    f"{expected_wpm:.0f} WPM, expected {WPM_MIN}-{WPM_MAX})"
                )
            elif expected_wpm > WPM_MAX:
                issues.append(
                    f"{prefix}: Word count too high for duration "
                    f"({scene.word_count} words / {scene.duration_minutes} min = "
                    f"{expected_wpm:.0f} WPM, expected {WPM_MIN}-{WPM_MAX})"
                )

        # Scene duration check
        if scene.duration_minutes > MAX_SCENE_MINUTES:
            issues.append(
                f"{prefix}: Duration exceeds {MAX_SCENE_MINUTES} min "
                f"({scene.duration_minutes} min)"
            )

        # [MODE:] tag presence
        if "[MODE:" not in scene.script_full:
            issues.append(f"{prefix}: Missing [MODE:] tag")
        else:
            mode_values = re.findall(r"\[MODE:\s*([^\]]+)\]", scene.script_full)
            invalid_modes = [mode for mode in mode_values if mode.strip() not in ALLOWED_MODES]
            if invalid_modes:
                issues.append(
                    f"{prefix}: Invalid [MODE:] value(s): {', '.join(sorted(set(invalid_modes)))}"
                )

        # [VISUAL:] tag for animation modes
        if "[MODE: animation]" in scene.script_full or "[MODE: overlay]" in scene.script_full:
            if "[VISUAL:" not in scene.script_full:
                issues.append(
                    f"{prefix}: Has animation/overlay mode but no [VISUAL:] tag"
                )

        # AI slop detection
        script_lower = scene.script_full.lower()
        for phrase in SLOP_PHRASES:
            if phrase in script_lower:
                issues.append(f"{prefix}: Contains AI slop phrase: \"{phrase}\"")

        # Empty script check
        if scene.word_count < 10:
            issues.append(f"{prefix}: Script too short ({scene.word_count} words)")

    # Total duration sanity check
    total = sum(s.duration_minutes for s in script.scenes)
    if script.total_minutes > 0 and abs(total - script.total_minutes) > 2.0:
        issues.append(
            f"Scene durations ({total:.1f} min) don't match total "
            f"({script.total_minutes:.1f} min)"
        )

    return issues
