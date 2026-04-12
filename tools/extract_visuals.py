"""Extract [VISUAL:] cues from production scripts.

Parses each scene's script_full to pull out visual descriptions along with
their surrounding context (active MODE, nearby narration). This gives the
image prompt engineer enough info to generate accurate, contextual images.
"""

from __future__ import annotations

import re

from tools.models import ProductionScript, VisualCue


# Matches [VISUAL: ...] — captures the description inside
_VISUAL_RE = re.compile(r"\[VISUAL:\s*([^\]]+)\]")
# Matches [MODE: ...] — captures the mode name
_MODE_RE = re.compile(r"\[MODE:\s*([^\]]+)\]")
# Strips all production tags to get surrounding narration
_TAG_RE = re.compile(r"\[(?:MODE|VISUAL|TEXT|AVATAR|PACE):[^\]]*\]")


def extract_visual_cues(script: ProductionScript) -> list[VisualCue]:
    """Extract all [VISUAL:] cues from a production script.

    Returns a flat list of VisualCue objects across all scenes, ordered
    by scene number then position within the scene.
    """
    cues: list[VisualCue] = []

    for scene in script.scenes:
        scene_num = _parse_scene_number(scene.scene)
        scene_title = scene.scene
        text = scene.script_full

        # Track the most recently seen [MODE:] as we scan left-to-right
        current_mode = "avatar"  # default if no MODE precedes the VISUAL

        # Find all tags in order to track mode context
        all_tags = list(re.finditer(r"\[(MODE|VISUAL):\s*([^\]]+)\]", text))

        cue_index = 0
        for match in all_tags:
            tag_type = match.group(1)
            tag_value = match.group(2).strip()

            if tag_type == "MODE":
                current_mode = tag_value
            elif tag_type == "VISUAL":
                # Get ~100 chars of narration around this visual cue
                surrounding = _get_surrounding_narration(text, match.start(), radius=150)

                cues.append(VisualCue(
                    scene_number=scene_num,
                    scene_title=scene_title,
                    cue_index=cue_index,
                    raw_description=tag_value,
                    mode=current_mode,
                    surrounding_narration=surrounding,
                ))
                cue_index += 1

    return cues


def _parse_scene_number(scene_label: str) -> int:
    """Extract the scene number from labels like '3 - The Mechanism'."""
    match = re.match(r"(\d+)", scene_label.strip())
    return int(match.group(1)) if match else 0


def _get_surrounding_narration(text: str, position: int, radius: int = 150) -> str:
    """Get clean narration text around a position, stripping all tags."""
    start = max(0, position - radius)
    end = min(len(text), position + radius)
    chunk = text[start:end]
    clean = _TAG_RE.sub("", chunk)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean
