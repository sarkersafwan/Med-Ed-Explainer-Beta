"""Character sheet builder for patient consistency across segments.

Flow:
  1. Decide if a character is needed based on segment intents.
  2. Derive a canonical CharacterSpec from PDF vignette and/or human-intent
     narration chunks.
  3. Generate a MULTI-ANGLE reference sheet via Nano Banana Pro (Gemini 3 Pro
     Image) that will be passed as a reference image to every clinical_scene /
     patient_experience segment downstream.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from tools.models import CharacterSpec, MedicalContent, ProductionScript, Segment
from tools.provider import chat_json

HUMAN_INTENTS = {"clinical_scene", "patient_experience"}


def character_is_needed(segments: list[Segment]) -> bool:
    """Smart gate: character is needed only if segments actually depict a patient."""
    return any(s.intent in HUMAN_INTENTS for s in segments)


def build_character_spec(
    content: MedicalContent | None,
    script: ProductionScript,
    segments: list[Segment] | None = None,
) -> CharacterSpec:
    """Derive a canonical character spec from the best available source.

    Priority: PDF clinical_vignette → human-intent narration chunks → neutral fallback.
    """
    vignette = (content.clinical_vignette if content else "") or ""
    human_chunks = " ".join(
        s.narration_chunk for s in (segments or []) if s.intent in HUMAN_INTENTS
    )
    hook_scenes = []
    for scene in script.scenes:
        label = scene.scene.lower()
        if "patient" in label or "question" in label:
            clean = re.sub(r"\[(?:MODE|VISUAL|TEXT|AVATAR|PACE):[^\]]*\]", "", scene.script_full)
            hook_scenes.append(clean.strip())
    source_text = "\n\n".join(part for part in [vignette, human_chunks, "\n\n".join(hook_scenes[:2])] if part).strip()

    if not source_text:
        # Last-resort neutral fallback
        return CharacterSpec(
            age="middle-aged",
            sex="male",
            ethnicity="unspecified",
            skin_tone="medium skin tone",
            build="average build",
            hair="short dark hair",
            facial_features="kind eyes, defined jawline",
            accessories="none",
            wardrobe="plain hospital gown",
            demeanor="tired, slightly anxious",
            continuity_notes="Keep the same age, face shape, skin tone, hairline, and wardrobe in every human shot.",
            one_line="A middle-aged patient of unspecified ethnicity, medium skin tone, average build, short dark hair, plain hospital gown, kind eyes and a defined jawline, tired and slightly anxious.",
        )

    system = (
        "You extract a canonical character description of the PATIENT from a medical "
        "education vignette or script. Return ONLY JSON, no prose. Be faithful to the "
        "source — do not invent details that contradict it. When the source is silent "
        "on a field, make a neutral plausible choice (e.g. 'average build', 'short "
        "dark hair', 'plain button-down shirt and slacks'). The goal is a LOCKED "
        "description that every downstream image prompt can reuse so the same person "
        "appears in every shot."
    )
    user = f"""Source text (PDF vignette and/or narration where the patient appears):
---
{source_text}
---

Return JSON with these exact keys:
{{
  "age": "e.g. '56-year-old' — use the exact age if stated, otherwise a decade range",
  "sex": "male | female",
  "ethnicity": "e.g. 'Caucasian' | 'South Asian' | 'Black' | 'East Asian' | 'Hispanic' | 'unspecified'",
  "skin_tone": "brief visual skin tone descriptor to keep image generation stable",
  "build": "e.g. 'average build' | 'stocky' | 'slender' | 'overweight'",
  "hair": "e.g. 'short salt-and-pepper hair' | 'bald' | 'shoulder-length brown hair'",
  "facial_features": "2-5 stable facial descriptors such as face shape, eyes, brows, nose, beard, smile lines",
  "accessories": "stable accessories if present, otherwise 'none'",
  "wardrobe": "what they are wearing consistent with the setting (street clothes if arriving at ED, hospital gown if admitted)",
  "demeanor": "emotional state from the narration — pained, breathless, anxious, tired, stoic, etc.",
  "continuity_notes": "one short sentence of identity-lock instructions for image generation, focused on what must NOT drift between shots",
  "one_line": "a SINGLE sentence (max ~35 words) that combines all of the above into a canonical description that will be injected into every image prompt. Start with age, then sex, ethnicity, build, hair, wardrobe, demeanor."
}}

Return ONLY the JSON object."""

    data = chat_json(
        system,
        user,
        model="gpt-4o",
        max_tokens=512,
        temperature=0.2,
    )
    return CharacterSpec(
        age=data.get("age", ""),
        sex=data.get("sex", ""),
        ethnicity=data.get("ethnicity", ""),
        skin_tone=data.get("skin_tone", ""),
        build=data.get("build", ""),
        hair=data.get("hair", ""),
        facial_features=data.get("facial_features", ""),
        accessories=data.get("accessories", ""),
        wardrobe=data.get("wardrobe", ""),
        demeanor=data.get("demeanor", ""),
        continuity_notes=data.get("continuity_notes", ""),
        one_line=data.get("one_line", ""),
    )


def generate_character_sheet(
    spec: CharacterSpec,
    output_dir: Path,
    override_image: str = "",
) -> Path:
    """Generate (or copy) the character reference PNG.

    If override_image is provided, copy it into output_dir/character/character.png and skip
    Nano Banana Pro. Otherwise call Nano Banana Pro text-to-image with a studio
    multi-angle character sheet prompt and save the result.
    """
    character_dir = output_dir / "character"
    character_dir.mkdir(parents=True, exist_ok=True)
    character_path = character_dir / "character.png"
    spec_path = character_dir / "character.json"

    if override_image:
        src = Path(override_image)
        if not src.exists():
            raise FileNotFoundError(f"--character-image not found: {override_image}")
        shutil.copyfile(src, character_path)
        print(f"    Using override character image: {src}")
    else:
        # Defer import to avoid circular dependency at module load
        from tools.generate_images import _generate_with_gemini

        image_bytes = _generate_with_gemini(_build_character_sheet_prompt(spec), reference_images=None)
        character_path.write_bytes(image_bytes)
        print(f"    Generated character sheet: {character_path}")

    spec.image_path = str(character_path)
    spec_path.write_text(json.dumps(spec.model_dump(), indent=2))
    return character_path


def _build_character_sheet_prompt(spec: CharacterSpec) -> str:
    """Build a multi-angle reference-sheet prompt for the patient character."""
    return (
        "Hyperreal multi-angle character reference sheet for a single patient, arranged as a clean six-view contact sheet on one canvas. "
        "Views required: front portrait, three-quarter left portrait, left profile, three-quarter right portrait, right profile, and full-body standing view. "
        "The exact same person must appear in every panel with identical age, ethnicity, face shape, hairline, wardrobe, accessories, and body proportions. "
        "Shot on Arri Alexa with 50mm lens, soft key light from camera left, subtle rim light, neutral dark charcoal studio background, crisp focus, "
        "prestige medical drama realism, honest non-stock human appearance, natural skin texture, hyperreal photographic detail. "
        f"Subject: {spec.one_line} "
        f"Skin tone: {spec.skin_tone}. Facial features: {spec.facial_features}. Accessories: {spec.accessories}. "
        f"Continuity lock: {spec.continuity_notes} "
        "Neutral relaxed expression, arms at sides in the full-body panel, no dramatic pose, no duplicated wardrobe changes, no stylization drift between panels. "
        "This is the ONLY context where a multi-panel layout is allowed. No text, no labels, no watermarks."
    )
