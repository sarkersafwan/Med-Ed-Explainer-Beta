"""Generate visual segments from scene scripts.

Breaks each scene into clip-length visual segments (one segment = one Kling
clip = one narrative beat), each with its own intent, hyperreal/cinematic
image prompt, and video motion prompt. The segment duration matches the
Kling clip length so the composed video stays aligned to the voice track.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

from tools.alignment import assign_segment_timings, validate_segment_coverage
from tools.models import CharacterSpec, ProductionScene, ProductionScript, Segment
from tools.provider import chat_text, get_text_model_name, parse_json_response

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", get_text_model_name())
PROMPTS_DIR = Path(__file__).parent.parent / "data" / "prompts"
# One segment = one Kling clip. Must match KLING_CLIP_SECONDS / animations.py.
CLIP_DURATION_SECONDS = int(os.environ.get("KLING_CLIP_SECONDS", "5"))
WORDS_PER_SECOND = 150 / 60  # 150 WPM baseline

VALID_INTENTS = {
    "clinical_scene",
    "patient_experience",
    "exam_or_imaging",
    "mechanism",
    "anatomy",
    "molecular",
    "comparison",
    "clinical_concept",
    "mechanism_summary",
    "data_or_concept",
}

HUMAN_INTENTS = {"clinical_scene", "patient_experience"}
BIOLOGY_INTENTS = {"mechanism", "anatomy", "molecular", "comparison", "exam_or_imaging", "mechanism_summary"}
INFORMATIVE_NONHUMAN_INTENTS = BIOLOGY_INTENTS | {"clinical_concept", "data_or_concept"}
SHORT_EXPLAINER_MINUTES = 1.0
CLINICAL_ARTIFACT_KEYWORDS = {
    "x-ray", "xray", "cxr", "ct", "mri", "ultrasound", "ekg", "ecg", "echo",
    "histology", "slide", "pathology", "specimen", "biopsy", "bronchoscope",
    "stethoscope", "chest tube", "needle", "catheter", "oxygen mask", "nasal cannula",
    "ventilator", "spirometer", "pleural kit", "procedure tray", "instrument tray",
    "drain", "syringe", "probe", "film", "scan",
}
INTERNAL_PATHOLOGY_KEYWORDS = {
    "pressure", "flow", "backs up", "collapse", "collapsed", "recoil", "air", "fluid",
    "pleural", "alveoli", "lung", "thoracic", "membrane", "seal", "vacuum", "edema",
    "ventricle", "atrium", "vein", "artery", "ischemia", "infarct", "obstruction",
    "pneumothorax", "tamponade", "effusion",
}
MOLECULAR_PROCESS_KEYWORDS = {
    "oxygen", "o2", "carbon dioxide", "co2", "diffusion", "diffuse",
    "channel", "ion", "ions", "sodium", "potassium", "calcium", "chloride",
    "proton", "receptor", "ligand", "binding", "binds", "enzyme", "substrate",
    "transporter", "pump", "atp", "adp", "mitochond", "electron transport",
    "hemoglobin", "heme", "surfactant", "alveolar-capillary", "capillary membrane",
    "synapse", "synaptic", "neurotransmitter", "vesicle", "signal transduction",
    "second messenger", "osmosis", "molecular", "microscopic", "subcellular",
}


def generate_segments(
    script: ProductionScript,
    style_direction: str = "",
    character: CharacterSpec | None = None,
) -> list[Segment]:
    """Generate visual segments for all scenes in a script.

    Each scene is broken into ~20-second segments based on word count.
    Each segment gets a hyperreal image prompt and a video motion prompt.

    Args:
        script: The production script with scenes.
        style_direction: Visual style guidance from creative brief / style reference.
    """
    image_system_prompt = (PROMPTS_DIR / "image_prompt.txt").read_text()
    video_system_prompt = (PROMPTS_DIR / "video_prompt.txt").read_text()
    if style_direction:
        image_system_prompt += f"\n\n## CLIENT STYLE DIRECTION\n{style_direction}\nApply this style to ALL image prompts."
    all_segments: list[Segment] = []

    for scene in script.scenes:
        scene_num = _parse_scene_number(scene.scene)
        segment_count = _calculate_target_count(scene)

        if segment_count == 0:
            continue

        print(f"    Scene {scene_num}: generating {segment_count} segments...")

        segments = _generate_scene_segments(
            image_system_prompt,
            video_system_prompt,
            scene,
            scene_num,
            segment_count,
            character=character,
            total_video_minutes=script.total_minutes,
        )
        all_segments.extend(segments)

    return all_segments


def _calculate_target_count(scene: ProductionScene) -> int:
    """Target one segment per Kling clip of spoken audio."""
    spoken_seconds = scene.word_count / WORDS_PER_SECOND
    effective_seconds = max(scene.duration_minutes * 60, spoken_seconds)
    return max(1, math.ceil(effective_seconds / CLIP_DURATION_SECONDS))


def _generate_scene_segments(
    image_system_prompt: str,
    video_system_prompt: str,
    scene: ProductionScene,
    scene_num: int,
    segment_count: int,
    max_retries: int = 2,
    character: CharacterSpec | None = None,
    total_video_minutes: float | None = None,
) -> list[Segment]:
    """Generate segments for a single scene using GPT."""
    min_count = max(1, segment_count - 1)
    max_count = segment_count + 1
    words_per_clip = max(1, round(CLIP_DURATION_SECONDS * WORDS_PER_SECOND))
    scene_purpose = _infer_scene_purpose(scene.scene)
    purpose_guidance = _scene_purpose_guidance(scene_purpose, segment_count)

    character_lock = ""
    if character and character.one_line:
        character_lock = (
            "\n## CHARACTER LOCK\n"
            "When a segment intent is clinical_scene or patient_experience, it must depict the SAME patient every time.\n"
            f"Canonical patient: {character.one_line}\n"
            f"Continuity notes: {character.continuity_notes or 'Keep age, face shape, skin tone, hair, and wardrobe stable across all human shots.'}\n"
            "Do not drift the face, ethnicity, age, hairline, or wardrobe between human shots unless narration explicitly requires a change.\n"
        )

    prompt = f"""Break this medical education video scene into visual segments. Each segment is ONE {CLIP_DURATION_SECONDS}-second Kling video clip that plays over the matching portion of narration audio. Aim for {segment_count} segments (allowed range: {min_count}–{max_count}). Each narration_chunk should cover roughly {words_per_clip} spoken words ({CLIP_DURATION_SECONDS}s at 150 WPM). Cut boundaries on sentences/beats, not mid-phrase.

**Scene:** {scene.scene}
**Duration:** {scene.duration_minutes} minutes
**Narration:**
{scene.script}
{character_lock}

## STEP 1 — identify narrative beats
Read the full narration. Every sentence belongs to exactly ONE of these beat types (the "intent"):
- clinical_scene — a patient/setting/scenario is being described (demographics, arrival, exam room, ambulance, home). Visual is a CINEMATIC HUMAN SCENE, not anatomy.
- patient_experience — what the patient feels/looks like (pain, gasping, wincing, weakness). Visual is a HUMAN CLOSE-UP, not anatomy.
- exam_or_imaging — a physical exam finding or imaging/lab result is described (CXR, CT, ECG, echo, murmur on auscultation, pitting edema, distended neck veins, crackles, rash, focal weakness). Visual is the MODALITY or focused exam detail itself.
- mechanism — pathophysiology/process described in motion (pressure backs up, ions flood, cells migrate). Hyperreal anatomy in action.
- anatomy — a static anatomical structure is introduced. Hyperreal cross-section.
- molecular — receptors, drugs, ions, proteins at molecular scale.
  Use this for biochemical or microscopic processes such as oxygen diffusion across the alveolar membrane, hemoglobin binding, ion-channel flow, receptor-ligand binding, vesicle release, transporter pumps, or ATP/mitochondrial mechanisms.
- comparison — healthy vs diseased, before/after.
- clinical_concept — a statistic, risk factor, or diagnostic framing beat shown through a real clinical artifact, pathology specimen, instrument, or grounded real-world setup. No metaphor. No anatomy unless the narration explicitly points there.
- mechanism_summary — a concept or teaching-summary beat that is still best shown through visceral in-body anatomy and pressure-flow context. This should feel like an internal documentary frame, not a plastic prop.
- data_or_concept — legacy alias only. If you were going to use this, choose either clinical_concept or mechanism_summary instead.

## STEP 2 — segment
- Narration chunks MUST be contiguous and together reproduce the full scene narration verbatim (no gaps, no overlap, no paraphrasing).
- One segment = one beat = one Kling clip. If a beat spans more than ~{words_per_clip * 2} words, split it into consecutive clips with the same intent.
- The FIRST segment of a scene that opens with a vignette MUST be intent=clinical_scene — do NOT jump straight to anatomy.
- Human shots are useful setup, but they are less informative than biology/mechanism shots. Once the patient setup is established, prefer mechanism, anatomy, exam_or_imaging, comparison, or other non-human shots whenever the narration allows it.
- For mechanism-heavy scenes, default to internal biology in action, not repeated character coverage.
- Avoid spending multiple consecutive clips on the character unless the narration is literally about what the patient is doing or feeling in that moment.
- Physical exam findings should usually become exam_or_imaging close-ups or focused clinical details, not full-body patient scenes.
- Avoid symbolic visuals such as storm clouds, city skylines, dams, traffic jams, floating isolated hearts, or surreal composites. Keep concepts grounded in medicine.
- In short medical explainers, most summary/concept beats about internal pathology should be mechanism_summary, not clinical_concept.
- This video is {total_video_minutes or scene.duration_minutes:.2f} minutes long. For explainers at or under {SHORT_EXPLAINER_MINUTES:.1f} minute, avoid clinical_concept unless the narration explicitly names a real clinical artifact, imaging study, or exam setup. For abstract internal pathology beats, use mechanism_summary instead.
- When narration describes a biochemical, molecular, microscopic, or cellular process, prefer intent=molecular and show the process at micro scale rather than zooming out to a whole-organ overview.
{purpose_guidance}

## STEP 3 — for each segment produce
1. segment_title — short descriptive title
2. intent — one of the enum values above
3. narration_chunk — the exact contiguous portion of narration this clip plays over
4. image_prompt — a prompt that LITERALLY depicts narration_chunk per its intent, following the matching branch of the image system prompt (HUMAN branch for clinical_scene/patient_experience, MODALITY branch for exam_or_imaging, ANATOMY branch for mechanism/anatomy/comparison/mechanism_summary, MOLECULAR branch for molecular, CONCEPT branch for clinical_concept). Do NOT use the anatomy template for human beats.
5. video_prompt — motion for the {CLIP_DURATION_SECONDS}-second clip that matches the intent (human action for clinical_scene/patient_experience; modality animation for exam_or_imaging; biology-in-motion for mechanism/anatomy/molecular/mechanism_summary; grounded clinical motion for clinical_concept). Max 500 chars.

## EXAMPLES (one per intent type)

clinical_scene
  narration_chunk: "A 56-year-old man walks into the ED clutching his chest, short of breath."
  image_prompt: "Cinematic film still, shot on Arri Alexa 35mm anamorphic, shallow depth of field, practical ED fluorescent lighting with cyan monitor glow. A 56-year-old man in a sweat-damp button-down grips his sternum with a clenched fist, face pale and tight with pain, stepping through the emergency department sliding doors. Triage nurse approaches from the right with a wheelchair. Desaturated teal-amber prestige drama color grade. Background: ED waiting area. No text."
  video_prompt: "Man staggers two steps through ED doors gripping his chest, jaw clenched, triage nurse rushes in from right with wheelchair, overhead fluorescents flicker, handheld camera drifts subtly."

patient_experience
  narration_chunk: "He describes a crushing pressure radiating down his left arm."
  image_prompt: "Cinematic extreme close-up, shot on Arri Alexa 35mm, shallow depth of field, monitor glow from below. A middle-aged man's face contorted in pain, eyes squeezed shut, a bead of sweat rolling down his temple, his right hand white-knuckled against his sternum. Desaturated teal-amber grade. Dark cinematic background. No text."
  video_prompt: "Slow-motion close-up: jaw clenches tighter, bead of sweat rolls down temple, fist presses harder into sternum, breath hitches sharply, subtle handheld drift."

exam_or_imaging
  narration_chunk: "His neck veins are distended and both legs show pitting edema."
  image_prompt: "Photorealistic focused clinical exam detail: distended neck veins under soft hospital lighting and bilateral pitting edema at the shins and ankles, realistic skin texture, subtle monitor glow, shallow depth of field, tight medical framing, no text, no labels."
  video_prompt: "Subtle respiratory rise and fall in the neck accentuates distended jugular veins, fingers press into the shin leaving a brief pitting indentation, gentle handheld clinical camera drift."

mechanism
  narration_chunk: "Pressure backs up into the pulmonary veins and fluid leaks into the alveoli."
  image_prompt: "Create a 16K ultra hyperrealistic cinematic rendering of a human thoracic cross-section showing engorged pulmonary veins and alveolar sacs filling with translucent edema fluid, captured with Canon EOS R5 85mm f/1.4, HDRI volumetric lighting, subsurface scattering on tissue, Octane render, sub-millimeter surface detail, dark cinematic background with volumetric god rays."
  video_prompt: "Pressure wave propagates backward through pulmonary veins which visibly engorge and bulge, translucent edema fluid seeps through capillary walls into alveolar spaces and pools, tissue glows faintly with inflammation."

molecular
  narration_chunk: "Oxygen diffuses across the alveolar membrane and binds hemoglobin in nearby red blood cells."
  image_prompt: "Create a 16K ultra hyperrealistic microscopic cinematic rendering of oxygen molecules diffusing across a wet alveolar-capillary membrane into nearby red blood cells, with phospholipid membrane detail, surfactant lining, plasma, hemoglobin-rich erythrocytes, volumetric light, subcellular texture, and documentary-grade biochemical realism. No labels, no abstract floating molecules in empty space."
  video_prompt: "Tiny oxygen molecules drift across the thin alveolar membrane into capillary plasma, slip into red blood cells, and bind hemoglobin in a slow luminous wave, with subtle Brownian motion and shallow microscopic camera drift."

clinical_concept
  narration_chunk: "Nearly one in three adults in the US has hypertension."
  image_prompt: "Hyperreal clinically grounded concept image of a primary care prep room lined with blood pressure cuffs, digital monitors, and chart trays, with multiple monitor screens showing elevated blood pressure values in soft focus, documentary-grade realism, dark cinematic background, no text overlays."
  video_prompt: "Monitor waveforms pulse softly, one blood pressure cuff slowly inflates around an arm on an exam chair, shallow camera push-in, practical clinic lighting, no surreal symbolism."

mechanism_summary
  narration_chunk: "Air in the pleural space breaks the vacuum seal, so the lung recoils inward and collapses."
  image_prompt: "Create a 16K ultra hyperrealistic cinematic rendering of an in-body thoracic cross-section showing pleural air separating the visceral pleura from the chest wall, with the lung recoiling inward and collapsing, wet pleural membranes, rib and intercostal context, visceral surgical-documentary realism, dark cinematic background."
  video_prompt: "Pleural air expands along the chest wall, the visceral pleura peels away, the lung recoils inward with elastic snap, subtle respiratory motion fades, slow documentary camera drift."

Return JSON array:
[
  {{
    "segment_title": "...",
    "intent": "clinical_scene",
    "narration_chunk": "...",
    "image_prompt": "...",
    "video_prompt": "..."
  }}
]

Return ONLY the JSON array."""

    combined_system = (
        image_system_prompt
        + "\n\n## VIDEO MOTION DIRECTION\n"
        + video_system_prompt
    )

    for attempt in range(max_retries + 1):
        try:
            raw = chat_text(
                combined_system,
                prompt,
                model=DEFAULT_MODEL,
                max_tokens=4096,
                temperature=0.3,
            )
            data = parse_json_response(raw)
        except Exception as e:
            if attempt == max_retries:
                raise ValueError(
                    f"Segment generation returned invalid JSON for scene {scene.scene}: {e}"
                ) from e
            prompt += (
                "\n\nCORRECTION:\n"
                "Your last response was not valid JSON.\n"
                "Return ONLY a valid JSON array with double-quoted keys and string values.\n"
                "Do not include markdown fences, comments, ellipses, or explanatory text."
            )
            continue

        if not isinstance(data, list):
            data = data.get("segments", [data])

        segments = []
        for i, seg in enumerate(data):
            raw_intent = str(seg.get("intent", "")).strip().lower()
            intent = _normalize_segment_intent(
                raw_intent,
                scene_purpose=scene_purpose,
                segment_title=str(seg.get("segment_title", "")),
                narration_chunk=str(seg.get("narration_chunk", "")),
                total_video_minutes=total_video_minutes,
            )
            segments.append(Segment(
                scene_number=scene_num,
                segment_index=i,
                segment_title=seg.get("segment_title", f"Segment {i+1}"),
                image_prompt=seg.get("image_prompt", ""),
                video_prompt=seg.get("video_prompt", ""),
                narration_chunk=seg.get("narration_chunk", ""),
                duration_seconds=float(CLIP_DURATION_SECONDS),
                intent=intent,
            ))

        if scene_purpose == "mechanism" and any(seg.intent in HUMAN_INTENTS for seg in segments):
            segments = _repair_mechanism_scene_segments(
                image_system_prompt,
                video_system_prompt,
                scene,
                scene_num,
                segments,
            )
        if _needs_short_explainer_concept_repair(segments, total_video_minutes):
            segments = _repair_short_explainer_concepts(
                image_system_prompt,
                video_system_prompt,
                scene,
                scene_num,
                segments,
                total_video_minutes=total_video_minutes,
            )

        coverage_issues = validate_segment_coverage(scene, segments)
        balance_issues = _validate_scene_visual_balance(
            scene,
            segments,
            total_video_minutes=total_video_minutes,
        )
        issues = coverage_issues + balance_issues
        if not issues:
            return assign_segment_timings(scene, segments)

        if attempt == max_retries:
            if not coverage_issues and scene_purpose != "mechanism":
                print(
                    f"      ⚠️  Accepting scene with balance warning(s) after retries: "
                    + "; ".join(balance_issues)
                )
                return assign_segment_timings(scene, segments)
            print("      ↺ Falling back to deterministic chunking for this scene...")
            fallback_segments = _fallback_segment_scene(
                image_system_prompt,
                video_system_prompt,
                scene,
                scene_num,
                segment_count,
                scene_purpose=scene_purpose,
                total_video_minutes=total_video_minutes,
            )
            fallback_issues = (
                validate_segment_coverage(scene, fallback_segments)
                + _validate_scene_visual_balance(
                    scene,
                    fallback_segments,
                    total_video_minutes=total_video_minutes,
                )
            )
            if not fallback_issues:
                return assign_segment_timings(scene, fallback_segments)
            raise ValueError(
                f"Segment coverage validation failed for scene {scene.scene}: {'; '.join(issues)}"
            )

        prompt += (
            "\n\nCORRECTION:\n"
            "Your last segmentation failed validation.\n"
            + "\n".join(f"- {issue}" for issue in issues)
            + "\nRewrite the FULL JSON array so the narration_chunk fields reconstruct the scene exactly."
        )

    return []


def _parse_scene_number(scene_label: str) -> int:
    match = re.match(r"(\d+)", scene_label.strip())
    return int(match.group(1)) if match else 0


def _normalize_segment_intent(
    raw_intent: str,
    *,
    scene_purpose: str,
    segment_title: str = "",
    narration_chunk: str = "",
    total_video_minutes: float | None = None,
) -> str:
    """Normalize model-returned intents and map the legacy concept bucket to clearer variants."""
    intent = (raw_intent or "").strip().lower()
    combined_text = f"{segment_title} {narration_chunk}".lower()

    if intent not in VALID_INTENTS:
        if _mentions_molecular_process(combined_text):
            return "molecular"
        return "mechanism"

    if intent == "molecular" and not _mentions_molecular_process(combined_text):
        if scene_purpose == "mechanism":
            return "mechanism"
        return "mechanism_summary"

    if intent not in HUMAN_INTENTS | {"exam_or_imaging"} and _mentions_molecular_process(combined_text):
        return "molecular"

    if intent == "data_or_concept":
        if _mentions_molecular_process(combined_text):
            return "molecular"
        if scene_purpose in {"mechanism", "takeaway"} or any(word in combined_text for word in INTERNAL_PATHOLOGY_KEYWORDS):
            return "mechanism_summary"
        return "clinical_concept"
    if intent == "clinical_concept" and _short_explainer_prefers_mechanism_summary(
        total_video_minutes=total_video_minutes,
        scene_purpose=scene_purpose,
        segment_title=segment_title,
        narration_chunk=narration_chunk,
    ):
        return "mechanism_summary"

    return intent


def _mentions_molecular_process(text: str) -> bool:
    """Return whether the narration is describing a biochemical or microscopic process."""
    lowered = (text or "").lower()
    return any(keyword in lowered for keyword in MOLECULAR_PROCESS_KEYWORDS)


def _infer_scene_purpose(scene_label: str) -> str:
    """Best-effort purpose inference from scene titles."""
    label = scene_label.lower()
    if "patient" in label:
        return "hook"
    if "question" in label:
        return "question"
    if "mechanism" in label:
        return "mechanism"
    if "why not" in label or "others" in label:
        return "differential"
    if "takeaway" in label:
        return "takeaway"
    return "general"


def _repair_mechanism_scene_segments(
    image_system_prompt: str,
    video_system_prompt: str,
    scene: ProductionScene,
    scene_num: int,
    segments: list[Segment],
) -> list[Segment]:
    """Ask the model to rewrite a mechanism scene so every clip stays non-human and biology-first."""
    repair_payload = [
        {
            "segment_index": seg.segment_index,
            "segment_title": seg.segment_title,
            "intent": seg.intent,
            "narration_chunk": seg.narration_chunk,
            "image_prompt": seg.image_prompt,
            "video_prompt": seg.video_prompt,
        }
        for seg in segments
    ]
    prompt = f"""Rewrite this mechanism scene so EVERY segment stays non-human and biology-first.

Scene: {scene.scene}
Narration:
{scene.script}

Current segments:
{json.dumps(repair_payload, indent=2)}

Rules:
- Keep the SAME number of segments.
- Keep each narration_chunk EXACTLY the same.
- Do NOT use clinical_scene or patient_experience anywhere.
- Prefer mechanism, anatomy, comparison, molecular, mechanism_summary, or exam_or_imaging.
- For symptom explanation beats inside a mechanism scene, use mechanism_summary rather than showing the patient's face.
- Make the image prompts visceral, in-body, and anatomically grounded.
- Avoid teaching mannequins, transparent plastic models, monitor-room props, visible text, overlay UI, or symbolic metaphors.

Return ONLY a JSON array with:
[
  {{
    "segment_title": "...",
    "intent": "mechanism",
    "narration_chunk": "...",
    "image_prompt": "...",
    "video_prompt": "..."
  }}
]"""

    combined_system = image_system_prompt + "\n\n## VIDEO MOTION DIRECTION\n" + video_system_prompt
    raw = chat_text(
        combined_system,
        prompt,
        model=DEFAULT_MODEL,
        max_tokens=4096,
        temperature=0.2,
    )
    data = parse_json_response(raw)
    if not isinstance(data, list):
        data = data.get("segments", [data])

    repaired: list[Segment] = []
    for i, seg in enumerate(data):
        raw_intent = str(seg.get("intent", "")).strip().lower()
        intent = _normalize_segment_intent(
            raw_intent,
            scene_purpose="mechanism",
            segment_title=str(seg.get("segment_title", "")),
            narration_chunk=str(seg.get("narration_chunk", "")),
        )
        repaired.append(Segment(
            scene_number=scene_num,
            segment_index=i,
            segment_title=seg.get("segment_title", f"Segment {i+1}"),
            image_prompt=seg.get("image_prompt", ""),
            video_prompt=seg.get("video_prompt", ""),
            narration_chunk=seg.get("narration_chunk", ""),
            duration_seconds=float(CLIP_DURATION_SECONDS),
            intent=intent,
        ))
    return repaired


def _repair_short_explainer_concepts(
    image_system_prompt: str,
    video_system_prompt: str,
    scene: ProductionScene,
    scene_num: int,
    segments: list[Segment],
    *,
    total_video_minutes: float | None,
) -> list[Segment]:
    """Rewrite text-prone short-form concept shots into mechanism summaries or focused artifacts."""
    repair_payload = [
        {
            "segment_index": seg.segment_index,
            "segment_title": seg.segment_title,
            "intent": seg.intent,
            "narration_chunk": seg.narration_chunk,
            "image_prompt": seg.image_prompt,
            "video_prompt": seg.video_prompt,
        }
        for seg in segments
    ]
    prompt = f"""This is a short medical explainer ({total_video_minutes or scene.duration_minutes:.2f} minutes total).
Rewrite any short-form concept segments so they do NOT rely on monitors, clipboards, dashboards, forms, or generic room props.

Scene: {scene.scene}
Narration:
{scene.script}

Current segments:
{json.dumps(repair_payload, indent=2)}

Rules:
- Keep the SAME number of segments.
- Keep each narration_chunk EXACTLY the same.
- For internal pathology or teaching-summary beats, prefer mechanism_summary shown through in-body anatomy.
- If the narration describes a biochemical, microscopic, or cellular process, use molecular instead and show the process at micro scale.
- Only keep clinical_concept if the narration explicitly calls for a real medical artifact or exam setup.
- Avoid monitors, clipboards, forms, dashboards, and text-bearing props.
- Prefer hyperreal anatomy, imaging, or focused exam detail that can be rendered without visible text.
- Keep the prompts hyperreal, medically grounded, and biology-first.

Return ONLY a JSON array with:
[
  {{
    "segment_title": "...",
    "intent": "mechanism_summary",
    "narration_chunk": "...",
    "image_prompt": "...",
    "video_prompt": "..."
  }}
]"""

    combined_system = image_system_prompt + "\n\n## VIDEO MOTION DIRECTION\n" + video_system_prompt
    raw = chat_text(
        combined_system,
        prompt,
        model=DEFAULT_MODEL,
        max_tokens=4096,
        temperature=0.2,
    )
    data = parse_json_response(raw)
    if not isinstance(data, list):
        data = data.get("segments", [data])

    repaired: list[Segment] = []
    for i, seg in enumerate(data):
        repaired.append(Segment(
            scene_number=scene_num,
            segment_index=i,
            segment_title=seg.get("segment_title", f"Segment {i+1}"),
            image_prompt=seg.get("image_prompt", ""),
            video_prompt=seg.get("video_prompt", ""),
            narration_chunk=seg.get("narration_chunk", ""),
            duration_seconds=float(CLIP_DURATION_SECONDS),
            intent=_normalize_segment_intent(
                str(seg.get("intent", "")).strip().lower(),
                scene_purpose=_infer_scene_purpose(scene.scene),
                segment_title=str(seg.get("segment_title", "")),
                narration_chunk=str(seg.get("narration_chunk", "")),
                total_video_minutes=total_video_minutes,
            ),
        ))
    return repaired


def _fallback_segment_scene(
    image_system_prompt: str,
    video_system_prompt: str,
    scene: ProductionScene,
    scene_num: int,
    segment_count: int,
    *,
    scene_purpose: str,
    total_video_minutes: float | None,
) -> list[Segment]:
    """Fallback path: lock chunk boundaries deterministically, then ask for shot design only."""
    chunks = _deterministic_narration_chunks(scene.script, segment_count)
    fixed_segments = [
        {
            "segment_index": i,
            "narration_chunk": chunk,
            "intent": _fallback_intent_for_chunk(
                chunk,
                index=i,
                total_chunks=len(chunks),
                scene_purpose=scene_purpose,
                total_video_minutes=total_video_minutes,
            ),
        }
        for i, chunk in enumerate(chunks)
    ]

    prompt = f"""You are repairing a failed segment generation. The narration chunks below are FIXED and already reconstruct the scene correctly.
Do NOT change narration_chunk or intent. Only provide a strong segment_title, image_prompt, and video_prompt for each fixed segment.

Scene: {scene.scene}
Narration:
{scene.script}

Fixed segments:
{json.dumps(fixed_segments, indent=2)}

Rules:
- Keep the SAME number of segments and the SAME segment_index values.
- Keep each narration_chunk EXACTLY identical.
- Keep each intent EXACTLY identical.
- Human intents must stay cinematic and character-consistent.
- Non-human intents must stay biology-first, hyperreal, and medically grounded.
- For short explainers, avoid monitors, clipboards, dashboards, forms, or text-prone props.
- Return ONLY a JSON array with keys: segment_index, segment_title, intent, narration_chunk, image_prompt, video_prompt.
"""

    combined_system = image_system_prompt + "\n\n## VIDEO MOTION DIRECTION\n" + video_system_prompt
    try:
        raw = chat_text(
            combined_system,
            prompt,
            model=DEFAULT_MODEL,
            max_tokens=4096,
            temperature=0.2,
        )
        data = parse_json_response(raw)
        if not isinstance(data, list):
            data = data.get("segments", [data])
        repaired: list[Segment] = []
        for i, seg in enumerate(data):
            fallback = fixed_segments[i] if i < len(fixed_segments) else fixed_segments[-1]
            repaired.append(Segment(
                scene_number=scene_num,
                segment_index=int(seg.get("segment_index", fallback["segment_index"])),
                segment_title=seg.get("segment_title", f"Segment {i+1}"),
                image_prompt=seg.get("image_prompt", _basic_image_prompt(fallback["narration_chunk"], fallback["intent"])),
                video_prompt=seg.get("video_prompt", _basic_video_prompt(fallback["narration_chunk"], fallback["intent"])),
                narration_chunk=str(seg.get("narration_chunk", fallback["narration_chunk"])),
                duration_seconds=float(CLIP_DURATION_SECONDS),
                intent=str(seg.get("intent", fallback["intent"])),
            ))
        return repaired
    except Exception:
        return [
            Segment(
                scene_number=scene_num,
                segment_index=item["segment_index"],
                segment_title=f"Segment {item['segment_index'] + 1}",
                image_prompt=_basic_image_prompt(item["narration_chunk"], item["intent"]),
                video_prompt=_basic_video_prompt(item["narration_chunk"], item["intent"]),
                narration_chunk=item["narration_chunk"],
                duration_seconds=float(CLIP_DURATION_SECONDS),
                intent=str(item["intent"]),
            )
            for item in fixed_segments
        ]


def _deterministic_narration_chunks(narration: str, target_count: int) -> list[str]:
    """Split narration into contiguous chunks that always reconstruct the original narration."""
    chunks = [chunk.strip() for chunk in re.findall(r"[^.!?]+[.!?]?", narration) if chunk.strip()]
    if not chunks:
        return [narration.strip()]

    while len(chunks) < target_count:
        longest_index = max(range(len(chunks)), key=lambda idx: len(chunks[idx]))
        split_parts = _split_chunk_for_fallback(chunks[longest_index])
        if not split_parts:
            break
        chunks = chunks[:longest_index] + split_parts + chunks[longest_index + 1:]

    if len(chunks) <= target_count:
        return chunks

    groups: list[str] = []
    remaining = list(chunks)
    slots_left = target_count
    while remaining and slots_left > 0:
        take = max(1, round(len(remaining) / slots_left))
        group = " ".join(part.strip() for part in remaining[:take] if part.strip()).strip()
        groups.append(group)
        remaining = remaining[take:]
        slots_left -= 1
    if remaining:
        groups[-1] = (groups[-1] + " " + " ".join(remaining)).strip()
    return groups


def _split_chunk_for_fallback(chunk: str) -> list[str] | None:
    """Split one narration chunk at a natural clause boundary if possible."""
    text = chunk.strip()
    comma_parts = [part.strip() for part in re.split(r",\s*", text) if part.strip()]
    if len(comma_parts) >= 2:
        midpoint = len(comma_parts) // 2
        left = ", ".join(comma_parts[:midpoint]).strip()
        right = ", ".join(comma_parts[midpoint:]).strip()
        return [left, right] if left and right else None

    words = text.split()
    if len(words) < 8:
        return None
    midpoint = len(words) // 2
    left = " ".join(words[:midpoint]).strip()
    right = " ".join(words[midpoint:]).strip()
    return [left, right] if left and right else None


def _fallback_intent_for_chunk(
    chunk: str,
    *,
    index: int,
    total_chunks: int,
    scene_purpose: str,
    total_video_minutes: float | None,
) -> str:
    """Pick a conservative, biology-first intent for a deterministic fallback chunk."""
    text = chunk.lower()
    if any(keyword in text for keyword in CLINICAL_ARTIFACT_KEYWORDS):
        return "exam_or_imaging"
    if _mentions_molecular_process(text):
        if scene_purpose == "hook" and index == 0:
            return "clinical_scene"
        return "molecular"
    if scene_purpose == "hook":
        if index == 0:
            return "clinical_scene"
        if index == 1 and total_chunks >= 3:
            return "patient_experience"
        return "mechanism_summary"
    if scene_purpose == "mechanism":
        return "mechanism"
    if scene_purpose in {"takeaway", "question", "general"}:
        if total_video_minutes is not None and total_video_minutes <= SHORT_EXPLAINER_MINUTES:
            return "mechanism_summary"
        return "clinical_concept"
    if any(keyword in text for keyword in INTERNAL_PATHOLOGY_KEYWORDS):
        return "mechanism_summary"
    return "mechanism"


def _basic_image_prompt(narration_chunk: str, intent: str) -> str:
    """Emergency fallback prompt if the repair model fails to return structured prompts."""
    if intent in HUMAN_INTENTS:
        return (
            "Cinematic film still, prestige medical drama realism. "
            f"Depict exactly this moment: {narration_chunk} "
            "Single coherent shot, realistic hospital lighting, no text."
        )
    if intent == "exam_or_imaging":
        return (
            "Photorealistic focused clinical detail or imaging artifact showing: "
            f"{narration_chunk} No text, no labels, tight framing."
        )
    if intent == "molecular":
        return (
            "Hyperreal microscopic biochemical render depicting: "
            f"{narration_chunk} Show the process at molecular or cellular scale with membranes, receptors, channels, nearby fluid, and tissue microenvironment visible. "
            "No abstract floating symbols, no text."
        )
    return (
        "Hyperreal in-body medical anatomy render depicting: "
        f"{narration_chunk} Keep the anatomy in situ with surrounding tissues, chest or body context visible, no text."
    )


def _basic_video_prompt(narration_chunk: str, intent: str) -> str:
    """Emergency fallback motion prompt if the repair model fails."""
    if intent in HUMAN_INTENTS:
        return f"Documentary-style human motion matching this beat: {narration_chunk}"
    if intent == "exam_or_imaging":
        return f"Subtle clinical artifact motion illustrating this finding: {narration_chunk}"
    if intent == "molecular":
        return (
            "Microscopic biochemical motion illustrating this process with slow, legible diffusion, binding, channel flow, "
            f"or transport: {narration_chunk}"
        )
    return f"Slow biology-first motion illustrating this mechanism: {narration_chunk}"


def _scene_purpose_guidance(purpose: str, segment_count: int) -> str:
    """Return purpose-specific guidance to bias the segmenter toward biology-first visuals."""
    if purpose == "hook":
        return (
            f"- Hook scene budget: at most {min(2, max(1, segment_count - 1))} human-intent clips. "
            "After the opening patient setup, pivot to focused exam details, imaging, or mechanism visuals as soon as the narration allows."
        )
    if purpose == "question":
        return (
            "- Question scenes should almost never linger on the patient. Prefer zero human-intent clips unless the narration explicitly describes the patient's body language in that exact beat."
        )
    if purpose == "mechanism":
        return (
            "- Mechanism scenes should be entirely biology-first: in-situ anatomy, pressure flow, edema, remodeling, cellular or vascular changes. Do not spend clips on the patient's face or body. If a teaching-summary beat is needed, use mechanism_summary rather than a human shot.\n"
            "- When the narration gets biochemical or microscopic, switch to molecular shots with membranes, receptors, diffusion, channel flow, vesicles, or hemoglobin/gas exchange detail rather than staying at the whole-organ level."
        )
    if purpose == "differential":
        return (
            "- Differential scenes should use comparative anatomy, imaging, labs, or pathology rather than repeated patient coverage."
        )
    if purpose == "takeaway":
        return (
            "- Takeaway scenes should summarize with mechanism, comparison, or treatment-relevant concept visuals. Avoid returning to the patient unless absolutely necessary."
        )
    return (
        "- Bias toward medically informative visuals over character coverage whenever both would be faithful."
    )


def _validate_scene_visual_balance(
    scene: ProductionScene,
    segments: list[Segment],
    *,
    total_video_minutes: float | None = None,
) -> list[str]:
    """Encourage biology-first shot choices and limit repeated character coverage."""
    issues: list[str] = []
    purpose = _infer_scene_purpose(scene.scene)
    human_count = sum(seg.intent in HUMAN_INTENTS for seg in segments)
    biology_count = sum(seg.intent in BIOLOGY_INTENTS for seg in segments)
    informative_nonhuman_count = sum(seg.intent in INFORMATIVE_NONHUMAN_INTENTS for seg in segments)

    max_human_by_purpose = {
        # Patient-introduction scenes often need a few human beats up front,
        # but we still want the majority of the scene to pivot into biology.
        "hook": min(2, max(1, len(segments) - 1)),
        "question": 0,
        "mechanism": 0,
        "differential": 0,
        "takeaway": 0,
        "general": max(1, len(segments) // 4),
    }
    max_human = max_human_by_purpose.get(purpose, 1)

    if human_count > max_human:
        issues.append(
            f"Scene '{scene.scene}' uses too many human-intent shots ({human_count}); target at most {max_human}"
        )

    if purpose == "mechanism" and biology_count == 0:
        issues.append(
            f"Scene '{scene.scene}' is a mechanism scene and should contain biology-first visuals"
        )

    if purpose in {"hook", "question", "takeaway"} and len(segments) >= 3 and biology_count == 0:
        if informative_nonhuman_count > 0:
            return issues
        issues.append(
            f"Scene '{scene.scene}' should mix in at least one biology/imaging/concept visual instead of all character coverage"
        )

    if total_video_minutes is not None and total_video_minutes <= SHORT_EXPLAINER_MINUTES:
        disallowed_short_concepts = [
            seg.segment_title for seg in segments
            if seg.intent == "clinical_concept" and not _clinical_concept_allowed(seg, total_video_minutes)
        ]
        if disallowed_short_concepts:
            issues.append(
                "Short explainers should avoid text-prone clinical_concept shots unless the narration explicitly names a real artifact: "
                + ", ".join(disallowed_short_concepts)
            )

    return issues


def _needs_short_explainer_concept_repair(
    segments: list[Segment],
    total_video_minutes: float | None,
) -> bool:
    """Return whether a short explainer still contains concept shots that should become anatomy-first."""
    return any(
        seg.intent == "clinical_concept" and not _clinical_concept_allowed(seg, total_video_minutes)
        for seg in segments
    )


def _clinical_concept_allowed(seg: Segment, total_video_minutes: float | None) -> bool:
    """Allow clinical_concept only when the narration explicitly anchors to a real artifact in short videos."""
    if total_video_minutes is None or total_video_minutes > SHORT_EXPLAINER_MINUTES:
        return True
    text = f"{seg.segment_title} {seg.narration_chunk} {seg.image_prompt}".lower()
    return any(keyword in text for keyword in CLINICAL_ARTIFACT_KEYWORDS)


def _short_explainer_prefers_mechanism_summary(
    *,
    total_video_minutes: float | None,
    scene_purpose: str,
    segment_title: str,
    narration_chunk: str,
) -> bool:
    """Bias short explainers toward anatomy-based summaries instead of generic concept props."""
    if total_video_minutes is None or total_video_minutes > SHORT_EXPLAINER_MINUTES:
        return False
    text = f"{segment_title} {narration_chunk}".lower()
    if any(keyword in text for keyword in CLINICAL_ARTIFACT_KEYWORDS):
        return False
    if scene_purpose in {"mechanism", "takeaway", "question", "general", "hook"}:
        return True
    return any(keyword in text for keyword in INTERNAL_PATHOLOGY_KEYWORDS)
