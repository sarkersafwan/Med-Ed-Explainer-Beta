"""Analyze medical content to determine scope, complexity, and teaching plan.

This is deterministic Python logic — no LLM calls. It creates the structural
skeleton that constrains and guides script generation.
"""

from __future__ import annotations

import re

from tools.models import DurationOption, MedicalContent, SceneBrief, TeachingPlan

SHORT_FORM_TARGET_MINUTES = 1.5


def analyze_content(content: MedicalContent) -> TeachingPlan:
    """Analyze extracted medical content and produce a teaching plan."""
    # Count key teaching elements
    core_concepts = _identify_core_concepts(content)
    differentials = _identify_differentials(content)
    chain_steps = _count_mechanism_steps(content.pathophysiology)

    # Complexity score (1-5)
    complexity = min(5, max(1, (
        (1 if chain_steps <= 2 else 2 if chain_steps <= 4 else 3)
        + (1 if len(differentials) >= 3 else 0)
        + (1 if len(core_concepts) >= 4 else 0)
    )))

    # Duration calculations
    # Base: 3 min for hook + question + takeaway
    # +1 min per mechanism step (these need visual explanation)
    # +0.5 min per differential to address
    base_minutes = 3.0
    mechanism_minutes = chain_steps * 1.0
    differential_minutes = len(differentials) * 0.5

    recommended = base_minutes + mechanism_minutes + differential_minutes
    minimum = base_minutes + min(mechanism_minutes, 2.0)  # Core mechanism only
    deep_dive = recommended + len(differentials) * 0.5 + 2.0  # Extra depth + clinical pearls

    duration_options = [
        DurationOption(
            label="recommended",
            minutes=round(recommended, 1),
            scene_count=_scenes_for_duration(recommended),
            description=f"Covers all {len(core_concepts)} core concepts and {len(differentials)} differentials",
        ),
        DurationOption(
            label="minimum",
            minutes=round(minimum, 1),
            scene_count=_scenes_for_duration(minimum),
            description="Core mechanism only — hook, pathophysiology, takeaway",
        ),
        DurationOption(
            label="deep_dive",
            minutes=round(deep_dive, 1),
            scene_count=_scenes_for_duration(deep_dive),
            description="Full coverage with clinical pearls, all differentials explained in depth",
        ),
    ]

    # Build scene briefs for recommended duration
    scenes = _build_scene_briefs(content, core_concepts, differentials, recommended)

    return TeachingPlan(
        topic=content.topic,
        complexity_score=complexity,
        concept_count=len(core_concepts),
        differential_count=len(differentials),
        recommended_minutes=round(recommended, 1),
        duration_options=duration_options,
        narrative_hook=_build_hook_summary(content),
        tension_point=content.question_stem or "What's causing this?",
        core_concepts=core_concepts,
        differential_concepts=[d for d in differentials],
        clinical_pearl=content.educational_objective or content.bottom_line,
        scenes=scenes,
    )


def rebuild_scenes_for_duration(
    content: MedicalContent,
    plan: TeachingPlan,
    target_minutes: float,
) -> list[SceneBrief]:
    """Rebuild scene briefs for a specific target duration."""
    return _build_scene_briefs(
        content,
        plan.core_concepts,
        plan.differential_concepts,
        target_minutes,
    )


def _identify_core_concepts(content: MedicalContent) -> list[str]:
    """Identify the core pathophysiology concepts to teach."""
    concepts = []

    # Parse pathophysiology into mechanism steps
    if content.pathophysiology:
        sentences = re.split(r"(?<=[.])\s+", content.pathophysiology)
        for s in sentences:
            s = s.strip()
            if len(s) > 20:  # Skip short fragments
                # Extract the key mechanism described
                concepts.append(s)

    # If we got nothing from pathophysiology, try key_info
    if not concepts and content.key_info:
        sentences = re.split(r"(?<=[.])\s+", content.key_info)
        concepts = [s.strip() for s in sentences if len(s.strip()) > 20]

    return concepts


def _identify_differentials(content: MedicalContent) -> list[str]:
    """Identify differential diagnoses / wrong answer concepts."""
    if content.wrong_answer_explanations:
        # Deduplicate by letter (A and E might share an explanation)
        seen = set()
        differentials = []
        for w in content.wrong_answer_explanations:
            if w.explanation not in seen:
                seen.add(w.explanation)
                label = f"{w.letter}. {w.text}" if w.text else f"Choice {w.letter}"
                differentials.append(label)
        return differentials

    # Fallback: non-correct answer choices
    return [
        f"{c.letter}. {c.text}"
        for c in content.answer_choices
        if c.letter != content.correct_answer_letter
    ]


def _count_mechanism_steps(pathophysiology: str) -> int:
    """Count the number of distinct mechanism steps in the pathophysiology chain."""
    if not pathophysiology:
        return 1
    sentences = re.split(r"(?<=[.])\s+", pathophysiology)
    return max(1, len([s for s in sentences if len(s.strip()) > 20]))


def _scenes_for_duration(minutes: float) -> int:
    """Estimate scene count for a target duration (~2 min per scene)."""
    return max(3, round(minutes / 2.0))


def _build_scene_briefs(
    content: MedicalContent,
    core_concepts: list[str],
    differentials: list[str],
    target_minutes: float,
) -> list[SceneBrief]:
    """Build ordered scene briefs for a target duration."""
    if target_minutes <= SHORT_FORM_TARGET_MINUTES:
        return _build_short_form_scene_briefs(
            content,
            core_concepts,
            target_minutes,
        )

    scenes: list[SceneBrief] = []
    scene_num = 1
    remaining_minutes = target_minutes

    # For very short videos, use proportional allocation
    min_scene_minutes = 0.15 if target_minutes < 1.0 else 0.5

    # Scene 1: Hook — open with a VISUAL, never a talking head.
    # If the source material provides a real clinical vignette (PDF/grounded
    # input), the hook is a cinematic patient shot that uses the character
    # sheet. Otherwise (topic-only / explainer input), the hook is a dramatic
    # anatomy/mechanism shot. In both cases: visual_mode is animation-
    # dominant so the script generator opens in [MODE: animation], not
    # [MODE: avatar]. Avatar PIP can still overlay the first beat.
    has_vignette = bool(content.clinical_vignette and content.clinical_vignette.strip())
    hook_mins = max(min_scene_minutes, min(1.5, remaining_minutes * 0.2))
    if has_vignette:
        hook_title = "The Patient"
        hook_content = content.clinical_vignette[:200]
    else:
        hook_title = "The Hook"
        # Prefer a concept-level teaser over a patient story for topic input.
        if content.core_concepts:
            hook_content = content.core_concepts[0]
        elif content.educational_objective:
            hook_content = content.educational_objective
        else:
            hook_content = content.topic
    scenes.append(SceneBrief(
        scene_number=scene_num,
        scene_title=hook_title,
        purpose="hook",
        key_content=hook_content,
        estimated_minutes=round(hook_mins, 1),
        visual_mode="animation_dominant",
    ))
    remaining_minutes -= hook_mins
    scene_num += 1

    # Scene 2: The clinical puzzle
    if content.question_stem and remaining_minutes > 1.0:
        puzzle_mins = min(1.5, remaining_minutes * 0.15)
        scenes.append(SceneBrief(
            scene_number=scene_num,
            scene_title="The Question",
            purpose="question",
            key_content=content.question_stem,
            estimated_minutes=round(puzzle_mins, 1),
            visual_mode="mixed",
        ))
        remaining_minutes -= puzzle_mins
        scene_num += 1

    # Middle scenes: Mechanism teaching (pathophysiology)
    # Allocate ~60% of remaining time
    mechanism_budget = remaining_minutes * 0.6
    if core_concepts:
        # Group concepts into scenes (2-3 concepts per scene)
        concepts_per_scene = max(1, len(core_concepts) // max(1, round(mechanism_budget / 2.0)))
        concept_groups = [
            core_concepts[i : i + concepts_per_scene]
            for i in range(0, len(core_concepts), concepts_per_scene)
        ]
        for group in concept_groups:
            scene_mins = min(3.0, mechanism_budget / len(concept_groups))
            scenes.append(SceneBrief(
                scene_number=scene_num,
                scene_title=f"The Mechanism — Part {scene_num - 2}" if len(concept_groups) > 1 else "The Mechanism",
                purpose="mechanism",
                key_content=" ".join(group)[:200],
                estimated_minutes=round(scene_mins, 1),
                visual_mode="animation_dominant",
            ))
            remaining_minutes -= scene_mins
            scene_num += 1

    # Differentials scene (if time allows)
    if differentials and remaining_minutes > 1.5:
        diff_mins = min(2.5, remaining_minutes * 0.5)
        scenes.append(SceneBrief(
            scene_number=scene_num,
            scene_title="Why Not the Others?",
            purpose="differential",
            key_content="; ".join(differentials)[:200],
            estimated_minutes=round(diff_mins, 1),
            visual_mode="mixed",
        ))
        remaining_minutes -= diff_mins
        scene_num += 1

    # Final scene: Clinical pearl / takeaway
    takeaway_mins = max(0.5, remaining_minutes)
    scenes.append(SceneBrief(
        scene_number=scene_num,
        scene_title="The Takeaway",
        purpose="takeaway",
        key_content=content.educational_objective or content.bottom_line or "Key learning point",
        estimated_minutes=round(takeaway_mins, 1),
        visual_mode="avatar_dominant",
    ))

    return scenes


def _build_short_form_scene_briefs(
    content: MedicalContent,
    core_concepts: list[str],
    target_minutes: float,
) -> list[SceneBrief]:
    """Build tightly bounded scene briefs for short-form explainers."""
    include_question = bool(content.question_stem and target_minutes >= 0.6)
    weights = [0.25, 0.5, 0.25]
    has_vignette = bool(content.clinical_vignette and content.clinical_vignette.strip())
    # Hook is ALWAYS visual — vignette-style patient shot when a real
    # vignette exists, dramatic anatomy/mechanism otherwise. Never avatar.
    hook_title = "The Patient" if has_vignette else "The Hook"
    hook_content = (
        content.clinical_vignette[:200]
        if has_vignette
        else (core_concepts[0] if core_concepts else content.topic)
    )
    scene_specs = [
        (hook_title, "hook", hook_content, "animation_dominant"),
        (
            "The Mechanism",
            "mechanism",
            " ".join(core_concepts)[:200] if core_concepts else (content.pathophysiology or "Core pathophysiology"),
            "animation_dominant",
        ),
        (
            "The Takeaway",
            "takeaway",
            content.educational_objective or content.bottom_line or "Key learning point",
            "avatar_dominant",
        ),
    ]

    if include_question:
        weights = [0.2, 0.15, 0.45, 0.2]
        scene_specs.insert(
            1,
            (
                "The Question",
                "question",
                content.question_stem,
                "mixed",
            ),
        )

    durations = _allocate_weighted_minutes(target_minutes, weights, precision=2)

    scenes: list[SceneBrief] = []
    for idx, (title, purpose, key_content, visual_mode) in enumerate(scene_specs, start=1):
        scenes.append(
            SceneBrief(
                scene_number=idx,
                scene_title=title,
                purpose=purpose,
                key_content=key_content,
                estimated_minutes=durations[idx - 1],
                visual_mode=visual_mode,
            )
        )

    return scenes


def _allocate_weighted_minutes(
    total_minutes: float,
    weights: list[float],
    *,
    precision: int,
) -> list[float]:
    """Allocate a duration budget exactly across weighted buckets."""
    if not weights:
        return []

    factor = 10 ** precision
    total_units = max(len(weights), int(round(total_minutes * factor)))
    total_weight = sum(weights) or float(len(weights))
    raw_units = [(weight / total_weight) * total_units for weight in weights]
    allocated_units = [int(units) for units in raw_units]
    remaining = total_units - sum(allocated_units)

    remainders = sorted(
        range(len(weights)),
        key=lambda idx: raw_units[idx] - allocated_units[idx],
        reverse=True,
    )
    for idx in remainders[:remaining]:
        allocated_units[idx] += 1

    allocations = [round(units / factor, precision) for units in allocated_units]
    correction = round(total_minutes - sum(allocations), precision)
    allocations[-1] = round(allocations[-1] + correction, precision)
    return allocations


def _build_hook_summary(content: MedicalContent) -> str:
    """Create a brief summary of the narrative hook from the vignette."""
    if not content.clinical_vignette:
        return ""
    # First sentence of the vignette
    m = re.match(r"(.+?[.])\s", content.clinical_vignette)
    return m.group(1) if m else content.clinical_vignette[:100]
