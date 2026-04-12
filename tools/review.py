"""Grounding artifacts and script review for production safety."""

from __future__ import annotations

import json
from pathlib import Path

from tools.models import (
    EvidenceSection,
    MedicalContent,
    ProductionScript,
    SceneBrief,
    SceneEvidence,
    ScriptReview,
)
from tools.provider import chat_json, get_text_model_name


def build_content_evidence(content: MedicalContent, grounded: bool) -> list[EvidenceSection]:
    """Convert extracted content into a normalized grounding bundle."""
    answer_choices = "\n".join(f"{c.letter}. {c.text}" for c in content.answer_choices)
    wrong_answers = "\n".join(
        f"{w.letter}. {w.text}: {w.explanation}" for w in content.wrong_answer_explanations
    )
    fields = [
        ("topic", "Topic", content.topic),
        ("subject", "Subject", content.subject),
        ("system", "System", content.system),
        ("clinical_vignette", "Clinical Vignette", content.clinical_vignette),
        ("question_stem", "Question Stem", content.question_stem),
        ("answer_choices", "Answer Choices", answer_choices),
        (
            "correct_answer",
            "Correct Answer",
            f"{content.correct_answer_letter}. {content.correct_answer}".strip(),
        ),
        ("pathophysiology", "Pathophysiology", content.pathophysiology),
        ("key_info", "Key Info", content.key_info),
        ("why_section", "Why Section", content.why_section),
        ("explanation", "Explanation", content.explanation),
        ("wrong_answer_explanations", "Wrong Answer Explanations", wrong_answers),
        (
            "educational_objective",
            "Educational Objective",
            content.educational_objective,
        ),
        ("bottom_line", "Bottom Line", content.bottom_line),
        ("diagram_description", "Diagram Description", content.diagram_description),
        (
            "diagram_labels",
            "Diagram Labels",
            ", ".join(content.diagram_labels),
        ),
    ]
    return [
        EvidenceSection(field_name=field_name, label=label, content=text, grounded=grounded)
        for field_name, label, text in fields
        if text
    ]


def build_scene_evidence(
    content: MedicalContent,
    scenes: list[SceneBrief],
    grounded: bool,
) -> list[SceneEvidence]:
    """Map each scene to the source sections it is allowed to use."""
    content_sections = {section.field_name: section for section in build_content_evidence(content, grounded)}

    mapping = {
        "hook": ["clinical_vignette", "topic", "subject", "system"],
        "question": ["question_stem", "answer_choices", "correct_answer"],
        "mechanism": ["pathophysiology", "why_section", "key_info", "diagram_description"],
        "differential": ["wrong_answer_explanations", "answer_choices", "explanation"],
        "takeaway": ["educational_objective", "bottom_line", "key_info"],
    }

    scene_evidence: list[SceneEvidence] = []
    for scene in scenes:
        allowed = mapping.get(scene.purpose, [])
        sections = [content_sections[key] for key in allowed if key in content_sections]
        scene_evidence.append(
            SceneEvidence(
                scene_number=scene.scene_number,
                scene_title=scene.scene_title,
                purpose=scene.purpose,
                source_sections=sections,
            )
        )
    return scene_evidence


def review_script_against_evidence(
    script: ProductionScript,
    scene_evidence: list[SceneEvidence],
    *,
    grounded: bool,
    pipeline_mode: str,
) -> ScriptReview:
    """Run a factual/production review using scene-level evidence bundles."""
    if pipeline_mode == "production" and not grounded:
        return ScriptReview(
            approved=False,
            grounded=False,
            requires_human_review=True,
            summary="Production mode blocks ungrounded topic-generated content.",
            blockers=[
                "Production mode requires PDF-grounded source material or an explicitly reviewed source bundle.",
            ],
            warnings=[
                "Switch to creative mode for brainstorming runs, or provide PDF source material for production.",
            ],
        )

    review_payload = {
        "pipeline_mode": pipeline_mode,
        "grounded": grounded,
        "model": get_text_model_name(),
        "scenes": [scene.model_dump() for scene in script.scenes],
        "scene_evidence": [scene.model_dump() for scene in scene_evidence],
    }

    system_prompt = (
        "You are a meticulous medical script reviewer. Review each scene ONLY against the "
        "provided evidence sections. Flag unsupported claims, missing support, production risk, "
        "and whether human review is needed. Be conservative. Return ONLY JSON."
    )
    user_prompt = f"""Review this script package and return JSON with this exact schema:
{{
  "approved": true,
  "grounded": true,
  "requires_human_review": false,
  "summary": "short summary",
  "blockers": ["..."],
  "warnings": ["..."],
  "scenes": [
    {{
      "scene": "1 - The Patient",
      "approved": true,
      "supported": true,
      "needs_human_review": false,
      "summary": "short scene summary",
      "issues": [
        {{
          "severity": "warning",
          "category": "grounding|medical_accuracy|production|flow",
          "message": "...",
          "source_sections": ["clinical_vignette"]
        }}
      ]
    }}
  ]
}}

Rules:
- If any scene introduces facts not justified by its allowed source sections, that is a blocker in production mode.
- If grounded=false, do not treat the content as medically verified; require human review even if internally consistent.
- Keep issue counts tight and practical.
- Return ONLY the JSON.

Package:
{json.dumps(review_payload, indent=2)}"""

    raw = chat_json(system_prompt, user_prompt, max_tokens=4096, temperature=0.1)
    review = ScriptReview(**raw)
    if not grounded:
        review.grounded = False
        review.requires_human_review = True
        if review.approved and pipeline_mode == "creative":
            review.warnings.append(
                "Creative-mode script is internally reviewed but not medically grounded to external evidence."
            )
    return review


def write_review_artifacts(
    evidence_sections: list[EvidenceSection],
    scene_evidence: list[SceneEvidence],
    script_review: ScriptReview,
    *,
    output_dir: Path,
) -> None:
    """Persist evidence and review artifacts next to a run."""
    evidence_dir = output_dir / "evidence"
    review_dir = output_dir / "review"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    (evidence_dir / "content_evidence.json").write_text(
        json.dumps([item.model_dump() for item in evidence_sections], indent=2)
    )
    (evidence_dir / "scene_evidence.json").write_text(
        json.dumps([item.model_dump() for item in scene_evidence], indent=2)
    )
    (review_dir / "script_review.json").write_text(
        json.dumps(script_review.model_dump(), indent=2, default=str)
    )
