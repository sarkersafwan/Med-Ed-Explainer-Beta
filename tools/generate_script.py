"""Generate production-ready video scripts using OpenAI GPT.

Two-pass approach:
  Pass 1: Creative generation from extracted content + teaching plan
  Pass 2: Quality review and targeted refinement
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from tools.models import (
    MedicalContent,
    ProductionScene,
    ProductionScript,
    SceneBrief,
    TeachingPlan,
)
from tools.project_store import slugify_project_name
from tools.provider import (
    chat_json,
    chat_text_messages,
    get_text_model_name,
    parse_json_response,
)

PROMPTS_DIR = Path(__file__).parent.parent / "data" / "prompts"

# Default model — GPT-4o for best quality, can override via env
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", get_text_model_name())


def generate_script(
    content: MedicalContent,
    plan: TeachingPlan,
    target_minutes: float | None = None,
    scenes: list[SceneBrief] | None = None,
    creative_direction: str = "",
) -> ProductionScript:
    """Generate a full production script from medical content and teaching plan.

    Args:
        content: Extracted medical content from PDF
        plan: Teaching plan with structural skeleton
        target_minutes: Override duration (uses plan.recommended_minutes if None)
        scenes: Override scene briefs (uses plan.scenes if None)
        creative_direction: Voice/avatar/style direction from creative brief
    """
    target = target_minutes or plan.recommended_minutes
    scene_briefs = scenes or plan.scenes
    total_target_words = max(1, int(round(target * 150)))
    scene_word_targets = _allocate_scene_word_targets(scene_briefs, total_target_words)

    system_prompt = (PROMPTS_DIR / "system_prompt.txt").read_text()
    if creative_direction:
        system_prompt += f"\n\n## CREATIVE DIRECTION (from client brief)\n{creative_direction}"

    # Generate scene-by-scene for better word count adherence
    all_scenes_data = []
    previous_script = ""

    for i, brief in enumerate(scene_briefs):
        scene_num = i + 1
        target_words = scene_word_targets[i]
        print(f"  Scene {scene_num}/{len(scene_briefs)}: {brief.scene_title}...")

        scene_data = _generate_scene_with_retry(
            system_prompt, content, brief, scene_briefs,
            target, previous_script, target_words,
        )
        all_scenes_data.append(scene_data)
        previous_script = scene_data.get("script", "")

    # Assemble the full script
    speech_prompt = "Warm, confident delivery. Natural eyebrow motion, subtle emphasis on key terms, steady pacing with occasional pauses for impact."
    assembled = {
        "total_minutes": target,
        "total_word_count": sum(_count_words(s.get("script", "")) for s in all_scenes_data),
        "speech_prompt": speech_prompt,
        "scenes": all_scenes_data,
    }

    reviewed = _review_and_refine(system_prompt, assembled, content)
    _enforce_mode_variety(reviewed)
    return _build_production_script(reviewed, content=content, plan=plan)


def _enforce_mode_variety(script_data: dict) -> None:
    """Hard guarantee that scripts aren't 100% avatar mode AND that the
    hook scene never opens in avatar mode.

    Two rules enforced:
      1. The very first scene's FIRST [MODE:] tag must be animation or
         overlay. A talking head as the opening shot kills retention.
      2. At least one scene must be animation/overlay (no all-avatar).

    Edits `script_data["scenes"]` in place.
    """
    scenes = script_data.get("scenes", [])
    if not scenes:
        return

    # Rule 1: hook scene never opens in avatar mode.
    first = scenes[0]
    first_text = first.get("script", "") or first.get("script_full", "")
    first_mode_match = re.search(r"\[MODE:\s*(avatar|animation|overlay)\s*\]", first_text)
    if first_mode_match and first_mode_match.group(1) == "avatar":
        print(f"  ⚠️  Hook scene opened in avatar mode — force-flipping to animation "
              f"for retention")
        flipped = re.sub(
            r"\[MODE:\s*avatar\s*\]",
            "[MODE: animation]",
            first_text,
            count=1,
        )
        if "[VISUAL:" not in flipped:
            title = first.get("scene") or first.get("scene_title") or "the opening concept"
            visual_hint = (
                f"[VISUAL: Ultradetailed hyperrealistic in-body anatomy establishing "
                f"{title}. Dramatic cinematic lighting, rich wet tissue texture, "
                f"pulled in tight on the relevant structures.]"
            )
            flipped = f"{visual_hint} {flipped}"
        first["script"] = flipped
        if "script_full" in first:
            first["script_full"] = flipped

    def _has_visual_mode(s: dict) -> bool:
        text = s.get("script", "") or s.get("script_full", "")
        return "[MODE: animation]" in text or "[MODE: overlay]" in text

    visual_count = sum(1 for s in scenes if _has_visual_mode(s))
    if visual_count > 0:
        return

    # All avatar. Flip the middle scene (or scenes) to animation. Keep the
    # first scene as hook-avatar and the last as takeaway-avatar when possible.
    n = len(scenes)
    if n == 1:
        targets = [0]
    elif n == 2:
        targets = [1]  # flip the second to animation
    else:
        # Flip everything that isn't the first or last scene.
        targets = list(range(1, n - 1))

    print(f"  ⚠️  Script came back all-avatar — force-flipping scene(s) "
          f"{[t + 1 for t in targets]} to animation mode for visual variety")

    for idx in targets:
        scene = scenes[idx]
        original = scene.get("script", "")
        # Replace the first [MODE: avatar] tag with [MODE: animation] and
        # append a generic [VISUAL:] placeholder so downstream pipeline works.
        flipped = re.sub(
            r"\[MODE:\s*avatar\s*\]",
            "[MODE: animation]",
            original,
            count=1,
        )
        if "[VISUAL:" not in flipped:
            title = scene.get("scene") or scene.get("scene_title") or "this concept"
            visual_hint = (
                f"[VISUAL: Hyperreal in-body medical animation illustrating "
                f"{title}. Show the mechanism and anatomy in clear educational detail, "
                f"no text or UI overlays.]"
            )
            flipped = f"{visual_hint} {flipped}"
        scene["script"] = flipped
        if "script_full" in scene:
            scene["script_full"] = flipped


def _generate_scene_with_retry(
    system_prompt: str,
    content: MedicalContent,
    scene: SceneBrief,
    all_scenes: list[SceneBrief],
    total_minutes: float,
    previous_script: str,
    target_words: int,
    max_retries: int = 2,
) -> dict:
    """Generate a scene, retrying if word count is too low."""
    scene_prompt = _build_scene_prompt(
        content, scene, all_scenes, total_minutes, previous_script, target_words
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": scene_prompt},
    ]

    for attempt in range(max_retries + 1):
        raw = chat_text_messages(
            messages,
            model=DEFAULT_MODEL,
            max_tokens=4096,
        )
        parsed = parse_json_response(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Scene generation did not return a JSON object")
        scene_data = parsed

        # Count actual spoken words (tags stripped)
        actual_words = _count_words(scene_data.get("script", ""))
        min_acceptable = int(target_words * 0.75)

        if actual_words >= min_acceptable or attempt == max_retries:
            if actual_words < min_acceptable:
                print(f"    ⚠️  Still short after retries: {actual_words}/{target_words} words")
            return scene_data

        # Retry: append the short output and ask for expansion
        print(f"    Retry ({actual_words}/{target_words} words — expanding)...")
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": (
            f"Your scene only has {actual_words} spoken words. I need {target_words} words. "
            f"That's {target_words - actual_words} more words of narration needed. "
            f"Rewrite the COMPLETE scene with more detailed explanations, more analogies, "
            f"and fuller sentences. Keep all tags but add MORE spoken narration between them. "
            f"Return the full JSON again."
        )})

    return scene_data


def _build_scene_prompt(
    content: MedicalContent,
    scene: SceneBrief,
    all_scenes: list[SceneBrief],
    total_minutes: float,
    previous_script: str,
    target_words: int,
) -> str:
    """Build the prompt for generating a single scene."""
    tolerance_words = 4 if target_words < 40 else 15
    sentence_min = max(1, round(target_words / 12))
    sentence_max = max(sentence_min, round(target_words / 8))
    short_form_guidance = (
        "This is a micro-scene, so keep the narration compact, vivid, and natural.\n"
        if target_words < 40
        else ""
    )

    # Context about the full video structure
    scene_list = "\n".join(
        f"  {'>>>' if s.scene_number == scene.scene_number else '   '} "
        f"Scene {s.scene_number}: [{s.purpose}] {s.scene_title}"
        for s in all_scenes
    )

    # Source material relevant to this scene's purpose
    source_material = _get_source_for_purpose(content, scene)

    continuity = ""
    if previous_script:
        # Last 100 chars of previous scene for continuity
        clean_prev = re.sub(r"\[(?:MODE|VISUAL|TEXT|AVATAR|PACE):[^\]]*\]", "", previous_script)
        last_line = clean_prev.strip().split("\n")[-1].strip()
        continuity = f"\n**Previous scene ended with:** \"{last_line[-150:]}\"\nMake your opening flow naturally from this.\n"

    return f"""Write Scene {scene.scene_number} of a {total_minutes}-minute medical education video.

**This scene:** "{scene.scene_title}" — Purpose: {scene.purpose}
**Duration:** {scene.estimated_minutes} minutes
**Target word count:** {target_words} SPOKEN words (tags don't count)
**Visual mode:** {scene.visual_mode}
{continuity}
**Full video structure (you are writing the >>> scene):**
{scene_list}

## SOURCE MATERIAL

**Topic:** {content.topic}

{source_material}

## REQUIREMENTS

Write approximately {target_words} words of spoken narration (±{tolerance_words} words).
That means roughly {sentence_min}-{sentence_max} spoken sentences.
{short_form_guidance}Include [MODE:], [VISUAL:], [TEXT:], [AVATAR:], [PACE:] tags inline.
Tags do NOT count toward the {target_words} word target.

Return JSON:
{{
  "scene": "{scene.scene_number} - {scene.scene_title}",
  "duration_minutes": {scene.estimated_minutes},
  "word_count": {target_words},
  "script": "<the full scene script with all tags>",
  "visual_summary": "<1-2 sentence description of dominant visuals>"
}}

Return ONLY the JSON."""


def _get_source_for_purpose(content: MedicalContent, scene: SceneBrief) -> str:
    """Get the relevant source material for a scene's purpose."""
    if scene.purpose == "hook":
        return f"**Clinical Vignette:**\n{content.clinical_vignette}"
    elif scene.purpose == "question":
        choices = "\n".join(f"  {c.letter}. {c.text}" for c in content.answer_choices)
        return f"**Question:** {content.question_stem}\n\n**Choices:**\n{choices}\n\n**Correct:** {content.correct_answer_letter}. {content.correct_answer}"
    elif scene.purpose == "mechanism":
        return f"**Pathophysiology:**\n{content.pathophysiology}\n\n**Why:**\n{content.why_section}\n\n**Key Info:**\n{content.key_info}"
    elif scene.purpose == "differential":
        wrong = "\n".join(
            f"  {w.letter}. {w.text}: {w.explanation}"
            for w in content.wrong_answer_explanations
        )
        return f"**Wrong Answer Explanations:**\n{wrong}"
    elif scene.purpose == "takeaway":
        return f"**Educational Objective:**\n{content.educational_objective}\n\n**Bottom Line:**\n{content.bottom_line}"
    return f"**Full content:**\n{content.pathophysiology}"


def _build_generation_prompt(
    content: MedicalContent,
    plan: TeachingPlan,
    scenes: list[SceneBrief],
    target_minutes: float,
) -> str:
    """Build the user message for script generation."""
    target_words = int(target_minutes * 150)

    scene_outline = "\n".join(
        f"  Scene {s.scene_number}: [{s.purpose}] \"{s.scene_title}\" "
        f"— {s.estimated_minutes} min, {int(s.estimated_minutes * 150)} words, "
        f"visual mode: {s.visual_mode}\n"
        f"    Content focus: {s.key_content[:150]}"
        for s in scenes
    )

    # Build the wrong answer section
    wrong_answers = ""
    if content.wrong_answer_explanations:
        wrong_answers = "\n".join(
            f"  Choice {w.letter} ({w.text}): {w.explanation}"
            for w in content.wrong_answer_explanations
        )

    return f"""Generate a {target_minutes}-minute medical education video script ({target_words} total words at 150 WPM).

## SOURCE MATERIAL (use ONLY these facts — do not hallucinate)

**Topic:** {content.topic}
**Subject:** {content.subject} | **System:** {content.system}

**Clinical Vignette:**
{content.clinical_vignette}

**Question:** {content.question_stem}

**Answer Choices:**
{chr(10).join(f"  {c.letter}. {c.text}" for c in content.answer_choices)}

**Correct Answer:** {content.correct_answer_letter}. {content.correct_answer}

**Pathophysiology:**
{content.pathophysiology}

**Key Info:**
{content.key_info}

**Why (mechanism reasoning):**
{content.why_section}

**Wrong Answer Explanations:**
{wrong_answers}

**Educational Objective:**
{content.educational_objective}

**Diagram Description (for visual reference):**
{content.diagram_description}

**Diagram Labels:** {", ".join(content.diagram_labels)}

## SCENE STRUCTURE (follow this exactly)

Target: {target_minutes} minutes, {target_words} words, {len(scenes)} scenes

{scene_outline}

## INSTRUCTIONS

Write the complete script following the scene structure above. Each scene must:
1. Match its target word count (±10%)
2. Fulfill its stated purpose (hook, question, mechanism, differential, takeaway)
3. Use the visual mode indicated (avatar_dominant → more [MODE: avatar], animation_dominant → more [MODE: animation])
4. Include [MODE:], [VISUAL:], [TEXT:], [AVATAR:], and [PACE:] tags as specified in the system prompt
5. Flow naturally from the previous scene

## WORD COUNT — THIS IS THE MOST IMPORTANT REQUIREMENT

The TOTAL SPOKEN NARRATION across all scenes must be approximately {target_words} words.

Tags like [MODE:], [VISUAL:], [TEXT:], [AVATAR:], [PACE:] DO NOT count as spoken words.
Only the actual narration text that the voice actor reads aloud counts.

Here are the EXACT spoken word targets per scene:
{chr(10).join(f"  Scene {s.scene_number}: MUST have {int(s.estimated_minutes * 150)} spoken words (not counting tags)" for s in scenes)}

A 1.4-minute scene = 210 words of actual narration. That's roughly 15-20 sentences.
Write FULL paragraphs of spoken narration. Not summaries. Not bullet points.
This is a real script that a human will read aloud. Make it LONG and DETAILED.

If your narration for any scene is under 150 words, you have FAILED the task. Rewrite it longer."""


def _review_and_refine(
    system_prompt: str,
    script_data: dict,
    content: MedicalContent,
) -> dict:
    """Pass 2: Review the generated script and refine if needed."""
    review_prompt = f"""Review this medical education video script for quality. Check:

1. MEDICAL ACCURACY: Does every fact match the source material? Flag any hallucinations.
2. ENGAGEMENT: Would a tired medical student at 2am keep watching? Are there enough rhetorical questions, analogies, and micro-suspense moments?
3. PRODUCTION FEASIBILITY: Does every [MODE: animation] scene have a [VISUAL:] tag? Are visual descriptions specific enough to generate images from?
4. PACING: Does each scene's word count match its target duration at 150 WPM?
5. FLOW: Do scenes transition naturally? No abrupt topic jumps?
6. AI SLOP: Any instances of "Let's dive in", "In this video", "It's important to note", or similar banned phrases?
7. [MODE:] USAGE: MANDATORY visual variety. If the script has ZERO [MODE: animation] or [MODE: overlay] scenes, that is an automatic REJECT — you MUST rewrite at least half the scenes as [MODE: animation] with [VISUAL:] tags showing the mechanism/anatomy/pathophysiology. Avatar-only scripts are forbidden. Teaching content (how/why/mechanism) always goes in animation mode; only hooks, transitions, and takeaways should stay in avatar mode.

**Educational Objective (must be addressed):**
{content.educational_objective}

**Script to review:**
{json.dumps(script_data, indent=2)}

If the script is good, return it unchanged as JSON.
If there are issues, fix them and return the corrected JSON.
Return ONLY the JSON — no commentary."""

    response = chat_json(
        system_prompt,
        review_prompt,
        model=DEFAULT_MODEL,
        max_tokens=16384,
    )
    if not isinstance(response, dict):
        raise ValueError("Script review did not return a JSON object")
    return response


def _strip_tags(text: str) -> str:
    """Strip production tags from script text, leaving only spoken narration."""
    cleaned = re.sub(r"\[(?:MODE|VISUAL|TEXT|AVATAR|PACE):[^\]]*\]", "", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"  +", " ", cleaned)
    return cleaned.strip()


def _count_words(text: str) -> int:
    """Count words in the spoken narration (tags stripped)."""
    clean = _strip_tags(text)
    return len(clean.split())


def _allocate_scene_word_targets(
    scene_briefs: list[SceneBrief],
    total_target_words: int,
) -> list[int]:
    """Allocate the total word budget across scenes while keeping micro-scenes viable."""
    if not scene_briefs:
        return []

    scene_count = len(scene_briefs)
    minimum_words = 10 if total_target_words >= scene_count * 10 else max(1, total_target_words // scene_count)
    weights = [max(scene.estimated_minutes, 0.01) for scene in scene_briefs]
    total_weight = sum(weights) or float(scene_count)

    targets = [minimum_words] * scene_count
    remaining_words = max(0, total_target_words - (minimum_words * scene_count))
    raw_extras = [(weight / total_weight) * remaining_words for weight in weights]
    extras = [int(extra) for extra in raw_extras]

    for idx in range(scene_count):
        targets[idx] += extras[idx]

    leftover = remaining_words - sum(extras)
    remainders = sorted(
        range(scene_count),
        key=lambda idx: raw_extras[idx] - extras[idx],
        reverse=True,
    )
    for idx in remainders[:leftover]:
        targets[idx] += 1

    return targets


def _build_production_script(
    data: dict,
    content: MedicalContent,
    plan: TeachingPlan,
) -> ProductionScript:
    """Build a ProductionScript from the parsed JSON response."""
    scenes = []
    for s in data.get("scenes", []):
        script_full = s.get("script", "")
        script_clean = _strip_tags(script_full)
        word_count = _count_words(script_full)

        scenes.append(ProductionScene(
            scene=s.get("scene", ""),
            duration_minutes=s.get("duration_minutes", 0),
            word_count=word_count,
            script=script_clean,
            script_full=script_full,
            speech_prompt=data.get("speech_prompt", ""),
            visual_summary=s.get("visual_summary", ""),
        ))

    return ProductionScript(
        project_name=slugify_project_name(content.topic),
        topic=content.topic,
        total_minutes=data.get("total_minutes", sum(s.duration_minutes for s in scenes)),
        total_word_count=sum(s.word_count for s in scenes),
        speech_prompt=data.get("speech_prompt", ""),
        scenes=scenes,
        source_pdf="input.pdf",
        generation_model=DEFAULT_MODEL,
    )
