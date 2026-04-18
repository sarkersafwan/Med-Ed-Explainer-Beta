"""Generate hyperrealistic medical images from segments or visual cues.

Supports two input modes:
  1. Segments (preferred): image_prompt is already engineered during segment generation
  2. Visual cues (legacy): GPT engineers prompts from raw [VISUAL:] descriptions

Uses Gemini's image API. Defaults to Nano Banana Pro
(`gemini-3-pro-image-preview`) which supports reference-image conditioning for
character consistency. Falls back to the legacy Imagen `:predict` endpoint if
`GEMINI_IMAGE_MODEL` is set to an `imagen-*` model.
"""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path

import httpx

from tools.models import CharacterSpec, GeneratedImage, ImagePrompt, Segment, VisualCue
from tools.parallel import run_parallel, safe_print
from tools.provider import chat_json, get_text_model_name, parse_json_response, vision_text

PROMPTS_DIR = Path(__file__).parent.parent / "data" / "prompts"
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", get_text_model_name())

# Gemini image model — defaults to Nano Banana Pro for reference-image support.
GEMINI_IMAGE_MODEL = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-3-pro-image-preview")
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

# Intents that should receive the character reference image.
HUMAN_INTENTS = {"clinical_scene", "patient_experience"}
BIOLOGY_INTENTS = {"mechanism", "anatomy", "molecular", "comparison", "mechanism_summary"}
IMAGE_QA_MAX_ATTEMPTS = max(1, int(os.environ.get("IMAGE_QA_MAX_ATTEMPTS", "3")))
# Network retries for Gemini API calls — transient connection resets and
# read timeouts would otherwise silently kill an image permanently.
GEMINI_NETWORK_RETRIES = max(1, int(os.environ.get("GEMINI_NETWORK_RETRIES", "4")))
GEMINI_NETWORK_BACKOFF = float(os.environ.get("GEMINI_NETWORK_BACKOFF", "3.0"))
KNOWN_QA_ISSUES = {
    "visible_text",
    "overlay_ui",
    "transparent_mannequin",
    "plastic_teaching_model",
    "symbolic_metaphor",
    "floating_isolated_organ",
    "weak_biology",
    "wrong_intent",
    "collage_or_split_screen",
    "glassy_cgi_texture",
    "operative_photo_bias",
}
MOLECULAR_CONTEXT_KEYWORDS = {
    "oxygen", "o2", "carbon dioxide", "co2", "diffusion", "membrane", "ion", "channel",
    "receptor", "ligand", "binding", "enzyme", "substrate", "atp", "mitochond",
    "hemoglobin", "heme", "surfactant", "alveolar", "capillary", "synapse",
    "neurotransmitter", "vesicle", "pump", "transporter", "gradient", "osmosis",
}


def generate_images_from_segments(
    segments: list[Segment],
    output_dir: Path,
    skip_existing: bool = True,
    character: CharacterSpec | None = None,
) -> list[GeneratedImage]:
    """Generate images for all segments. Prompts are already engineered.

    If `character` is provided AND its image_path exists, segments whose intent
    is `clinical_scene` or `patient_experience` receive the character reference sheet
    as a reference image so the same face appears across all human shots.
    """
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir = output_dir / "rejected"

    character_bytes: bytes | None = None
    if character and character.image_path and Path(character.image_path).exists():
        character_bytes = Path(character.image_path).read_bytes()
        print(f"  Character reference loaded: {character.image_path}")

    prompt_records: list[dict[str, object]] = []
    for seg in segments:
        use_reference = character_bytes is not None and seg.intent in HUMAN_INTENTS
        effective_prompt = _tighten_segment_prompt(seg, seg.image_prompt)
        if use_reference and character and character.one_line:
            effective_prompt = _apply_character_lock(effective_prompt, character)
        prompt_records.append({
            "scene": seg.scene_number,
            "segment": seg.segment_index,
            "title": seg.segment_title,
            "intent": seg.intent,
            "prompt": seg.image_prompt,
            "effective_prompt": effective_prompt,
            "video_prompt": seg.video_prompt,
            "use_character_reference": use_reference,
        })

    # Save prompts for debugging
    prompts_path = output_dir / "image_prompts.json"
    prompts_path.write_text(json.dumps(prompt_records, indent=2))

    print(f"  Generating {len(segments)} images via {GEMINI_IMAGE_MODEL}...")

    def _one_image(seg: Segment, i: int) -> GeneratedImage | None:
        filename = f"scene{seg.scene_number}_seg{seg.segment_index}.png"
        filepath = images_dir / filename

        if skip_existing and filepath.exists():
            safe_print(f"    [seg {seg.scene_number}.{seg.segment_index}] skip (exists)")
            return _make_generated_image(seg, str(filepath))

        use_reference = bool(prompt_records[i]["use_character_reference"])
        refs = [character_bytes] if use_reference else None
        prompt = str(prompt_records[i]["effective_prompt"])

        ref_tag = " [+char]" if use_reference else ""
        safe_print(f"    [seg {seg.scene_number}.{seg.segment_index}] "
                   f"{seg.intent or '?'}: {seg.segment_title}{ref_tag}")

        final_bytes: bytes | None = None
        last_generated_bytes: bytes | None = None  # Always keep the last image even if QA fails
        last_error: Exception | None = None
        qa_attempts: list[dict[str, object]] = []

        for attempt in range(1, IMAGE_QA_MAX_ATTEMPTS + 1):
            try:
                image_bytes = _generate_with_gemini(prompt, reference_images=refs)
                last_generated_bytes = image_bytes  # Track last successful generation
                review = _review_segment_image(seg, image_bytes, prompt)
                if not review.get("approved", False):
                    issues_now = list(review.get("issues", []))
                    if not _requires_regeneration(seg, issues_now):
                        review["approved"] = True
                        review["soft_override"] = True
                if not review.get("approved", False):
                    archive_paths = _archive_rejected_attempt(
                        rejected_dir, seg, attempt, image_bytes, review, prompt,
                    )
                    review.update(archive_paths)
                qa_attempts.append(review)

                if review.get("approved", False):
                    final_bytes = image_bytes
                    break

                issues = list(review.get("issues", []))
                if attempt == IMAGE_QA_MAX_ATTEMPTS or not _requires_regeneration(seg, issues):
                    safe_print(f"      [seg {seg.scene_number}.{seg.segment_index}] "
                               f"⚠️ QA soft-reject (keeping image): {', '.join(issues) or 'unspecified'}")
                    # NEVER drop to black — use the last generated image even if QA didn't love it
                    final_bytes = image_bytes
                    break

                safe_print(f"      [seg {seg.scene_number}.{seg.segment_index}] "
                           f"↺ regen after QA: {', '.join(issues)}")
                prompt = _build_regeneration_prompt(prompt, seg, issues)
                prompt_records[i]["effective_prompt"] = prompt
            except Exception as e:
                last_error = e
                if attempt == IMAGE_QA_MAX_ATTEMPTS:
                    break

        prompt_records[i]["qa_attempts"] = qa_attempts

        # If QA approved or soft-rejected, we have final_bytes.
        # If all attempts threw exceptions, fall back to the last image we got.
        if final_bytes is None and last_generated_bytes is not None:
            safe_print(f"    ⚠️ seg {seg.scene_number}.{seg.segment_index} using last-resort image (all QA attempts failed)")
            final_bytes = last_generated_bytes

        if final_bytes is not None:
            filepath.write_bytes(final_bytes)
            safe_print(f"    ✓ seg {seg.scene_number}.{seg.segment_index} saved {filename}")
            return _make_generated_image(seg, str(filepath))

        if last_error:
            safe_print(f"    ✗ seg {seg.scene_number}.{seg.segment_index} failed: {last_error}")
        return None

    results = run_parallel(
        segments,
        _one_image,
        max_workers=int(os.environ.get("IMAGE_PARALLEL", "5")),
        label="images",
    )
    # Write prompt records once at the end (single write, no lock needed).
    prompts_path.write_text(json.dumps(prompt_records, indent=2))
    return [r.value for r in results if r.ok and r.value is not None]


def generate_images(
    cues: list[VisualCue],
    output_dir: Path,
    skip_existing: bool = True,
) -> list[GeneratedImage]:
    """Legacy: generate images from visual cues (used when segments aren't available)."""
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print("  Engineering image prompts...")
    prompts = engineer_prompts(cues)
    print(f"  Engineered {len(prompts)} prompts")

    prompts_path = output_dir / "image_prompts.json"
    prompts_path.write_text(json.dumps(
        [{"scene": p.cue.scene_number, "cue": p.cue.cue_index,
          "raw": p.cue.raw_description, "prompt": p.prompt,
          "negative": p.negative_prompt, "tags": p.style_tags}
         for p in prompts],
        indent=2,
    ))

    print("  Generating images via Gemini Imagen...")
    generated: list[GeneratedImage] = []

    for i, prompt in enumerate(prompts):
        filename = f"scene{prompt.cue.scene_number}_cue{prompt.cue.cue_index}.png"
        filepath = images_dir / filename

        if skip_existing and filepath.exists():
            print(f"    [{i+1}/{len(prompts)}] Skipping (exists): {filename}")
            generated.append(GeneratedImage(prompt=prompt, file_path=str(filepath)))
            continue

        print(f"    [{i+1}/{len(prompts)}] Generating: {filename}")
        try:
            image_bytes = _generate_with_gemini(prompt.prompt)
            filepath.write_bytes(image_bytes)
            generated.append(GeneratedImage(prompt=prompt, file_path=str(filepath)))
            print(f"    ✓ Saved {filename}")
        except Exception as e:
            print(f"    ✗ Failed: {e}")

        if i < len(prompts) - 1:
            time.sleep(1)

    return generated


def _make_generated_image(seg: Segment, file_path: str) -> GeneratedImage:
    """Create a GeneratedImage from a Segment."""
    cue = VisualCue(
        scene_number=seg.scene_number,
        scene_title=seg.segment_title,
        cue_index=seg.segment_index,
        raw_description=seg.image_prompt,
    )
    prompt = ImagePrompt(
        cue=cue,
        prompt=seg.image_prompt,
        negative_prompt=(
            "text, labels, annotations, watermarks, cartoon, flat illustration, white background, "
            "split screen, collage, diptych, comic panel, surreal landscape, floating organ"
        ),
        style_tags=["hyperrealistic", "cinematic", "medical"],
    )
    return GeneratedImage(prompt=prompt, file_path=file_path)


def _apply_character_lock(prompt: str, character: CharacterSpec) -> str:
    """Inject a stronger continuity lock ahead of the image prompt."""
    continuity_lock = character.continuity_notes or (
        "Keep the same face shape, skin tone, hairline, age, and wardrobe in every human shot."
    )
    return (
        "CHARACTER CONTINUITY LOCK:\n"
        "The subject is the exact same person shown in the reference image.\n"
        f"Canonical patient: {character.one_line}\n"
        f"Skin tone: {character.skin_tone or 'match the reference image exactly'}.\n"
        f"Facial features: {character.facial_features or 'match the reference image exactly'}.\n"
        f"Accessories: {character.accessories or 'none'}.\n"
        f"Continuity rule: {continuity_lock}\n"
        "Use the multi-angle reference sheet to keep ethnicity, age, facial structure, hairstyle, body proportions, and wardrobe stable.\n\n"
        f"{prompt}"
    )


_SENSITIVE_ANATOMY_KEYWORDS = {
    "breast", "nipple", "areola", "mammary", "lactat",
    "genital", "penis", "vagina", "vulva", "uterus", "uterine",
    "ovary", "ovarian", "scrotum", "testis", "testicle", "prostate",
    "cervix", "cervical os", "endometri", "fallopian",
    "anus", "rectum", "rectal",
    "surgery", "surgical", "incision", "scalpel",
    "wound", "dissect", "amputat",
}


def _is_sensitive_anatomy(text: str) -> bool:
    low = text.lower()
    return any(kw in low for kw in _SENSITIVE_ANATOMY_KEYWORDS)


def _tighten_segment_prompt(seg: Segment, prompt: str) -> str:
    """Add deterministic grounding guardrails so scene images stay medically useful."""
    shared = (
        "Render a single coherent ultradetailed hyperrealistic medical frame "
        "with one camera setup and one moment in time. Think high-end "
        "cinematic anatomical render with extreme micro-detail, rich wet "
        "tissue texture, dramatic volumetric lighting, shallow depth of "
        "field, and textbook-grade anatomical accuracy. Not a photograph — "
        "this is a next-generation medical illustration that reads as "
        "hyperrealistic without being a literal photo. "
        "No split-screen, no collage, no diptych, no comic-panel layout, "
        "no montage, and no duplicate subject."
    )
    # Sensitive anatomy (breast/genital/reproductive/surgical) stays hyperreal
    # but picks up explicit safety constraints so downstream classifiers on
    # Gemini and Kling don't flag it. Cross-sections and in-body views are
    # still allowed — just no exposed external nudity, no surgical fields,
    # no blood/gore.
    if _is_sensitive_anatomy(prompt):
        shared += (
            " Show the anatomy as a hyperreal in-body cross-section embedded in "
            "surrounding tissue, fascia, and vasculature — never an exposed "
            "external view, never an open surgical field, never bare skin "
            "in a sexual context. No nudity of primary/secondary sexual "
            "characteristics, no external nipples, no external genitalia, "
            "no blood, no gore, no wound cavity, no incision edges, no scalpels, "
            "no operative retractors. Frame as educational anatomy seen from "
            "inside the body, as if the skin and subcutaneous layers were "
            "simply not in frame. Keep the tissue ultradetailed and "
            "hyperrealistic with cinematic lighting, not a literal photograph."
        )
    anatomy_context = _expected_anatomy_context(seg, prompt) if seg.intent in BIOLOGY_INTENTS else ""
    micro_context = _expected_molecular_context(seg, prompt) if seg.intent == "molecular" else ""

    if seg.intent in HUMAN_INTENTS:
        guardrail = (
            "Keep it documentary-real and clinically grounded. One patient only, one angle only, no multi-panel composition."
        )
    elif seg.intent == "exam_or_imaging":
        guardrail = (
            "Show the literal clinical artifact or a tightly framed physical exam detail. "
            "Do not pull back to a generic full-body patient shot unless the narration explicitly requires it."
        )
    elif seg.intent == "molecular":
        guardrail = (
            "Show the process at microscopic or biochemical scale with ultra-detailed local context: membranes, receptors, channels, vesicles, dissolved gases, plasma, organelles, or extracellular fluid as appropriate. "
            "Keep the molecules embedded in a believable biological microenvironment rather than floating as abstract balls in empty space. "
            "Make it feel like an Ultra HD medical documentary still designed to animate beautifully: legible diffusion, binding, transport, gating, or conformational change. "
            "Prefer matte wet biological surfaces, lipid bilayers, protein texture, and microfluid realism over glossy CGI spheres, infographic iconography, or generic chemistry art. "
            "No text, no labels, no overlay UI, no symbolic metaphor, and no broad whole-organ overview unless the narration explicitly needs it."
        )
    elif seg.intent == "mechanism_summary":
        guardrail = (
            "Teach the concept through visceral in-body anatomy rather than a prop. "
            "Show pressure-flow context, pleural separation, collapse, distension, recoil, or surrounding tissues inside the thoracic cavity. "
            "If any earlier wording suggests monitors, clipboards, forms, dashboards, or room props, ignore that and teach the beat through anatomy instead. "
            "Prefer matte, moist, organic pleura and living tissue over shiny acrylic translucence or hollow-shell CGI gloss. "
            "No plastic teaching model, no transparent mannequin, no overlay UI, no text."
        )
    elif seg.intent in BIOLOGY_INTENTS:
        guardrail = (
            "Keep the anatomy in situ within the correct body region with surrounding tissues, vessels, ribs, pleura, and spatial context visible. "
            "Prefer visceral in-body surgical-documentary realism with wet membranes, recoil, compression, or pressure effects. "
            "Prefer matte, fibrous, living tissue surfaces over glossy acrylic, glassy CGI membranes, or translucent hollow-shell renders. "
            "No floating isolated organs, no transparent mannequins, no museum specimen look, no monitor-room teaching prop, no overlay UI, and no fantasy composition."
        )
    elif seg.intent in {"clinical_concept", "data_or_concept"}:
        guardrail = (
            "Ground the concept in believable medical reality through a text-free clinical artifact such as an unlabeled instrument tray, pathology specimen, oxygen delivery setup, sealed procedure kit, or exam setup. "
            "Avoid monitors, dashboards, forms, clipboards, and other text-prone props unless the narration explicitly names them. "
            "Do not use storm clouds, city skylines, dams, traffic jams, cosmic imagery, symbolic landscapes, plastic teaching models, or transparent mannequins."
        )
    else:
        guardrail = (
            "Keep the frame medically grounded, ultradetailed, and "
            "hyperrealistic with cinematic lighting — rich texture, dramatic "
            "depth, textbook-accurate anatomy, not a literal photograph."
        )

    if anatomy_context:
        guardrail += " " + anatomy_context
    if micro_context:
        guardrail += " " + micro_context

    return f"{prompt.rstrip()} {shared} {guardrail}".strip()


def _review_segment_image(seg: Segment, image_bytes: bytes, requested_prompt: str = "") -> dict[str, object]:
    """Review a generated image and flag common failure modes before accepting it."""
    system_prompt = (
        "You are a strict medical image QA reviewer. Return ONLY JSON. "
        "Evaluate whether a generated image matches the requested intent and whether it has any disqualifying issues."
    )
    user_text = f"""Review this generated medical image.

Intent: {seg.intent}
Title: {seg.segment_title}
Narration chunk: {seg.narration_chunk}
Requested image prompt: {requested_prompt[:1200]}

Return JSON:
{{
  "approved": true,
  "issues": ["visible_text"],
  "summary": "short explanation"
}}

Possible issues:
- visible_text
- overlay_ui
- transparent_mannequin
- plastic_teaching_model
- symbolic_metaphor
- floating_isolated_organ
- weak_biology
- wrong_intent
- collage_or_split_screen
- glassy_cgi_texture
- operative_photo_bias

Rules:
- mechanism / anatomy / molecular / comparison / mechanism_summary images must look like in-body medical realism, not a plastic teaching model or monitor-room prop.
- molecular images should depict a specific microscopic biological process in context, such as diffusion across a membrane, receptor binding, ion flow, vesicle release, hemoglobin gas exchange, or organelle-level activity, not generic floating molecules in empty space.
- Flag glassy_cgi_texture if tissues read as acrylic, hollow-shell, overly glossy, synthetic CGI, or unnaturally glass-like instead of matte wet organic anatomy.
- Flag operative_photo_bias if the frame feels like an intraoperative gore photo or specimen shot rather than a clear educational anatomy render.
- clinical_concept images must still be clinically grounded, not symbolic, and should not contain visible text.
- Human-intent images should not become collages or multi-panel sheets.
- Approve only if the image is strong enough to keep as a final pipeline asset.
"""
    try:
        raw = vision_text(
            system_prompt,
            user_text,
            image_bytes,
            "image/png",
            max_tokens=400,
        )
        data = parse_json_response(raw)
        if isinstance(data, dict):
            return _normalize_review_result(seg, data)
    except Exception as e:
        return {
            "approved": True,
            "issues": [],
            "summary": f"qa_unavailable:{e}",
        }

    return {
        "approved": True,
        "issues": [],
        "summary": "",
    }


def _requires_regeneration(seg: Segment, issues: list[str]) -> bool:
    """Return whether the detected QA issues should trigger a regeneration attempt."""
    issue_set = set(issues)
    # Relaxed QA: only regenerate on hard failures (text/overlays/collage/wrong intent
    # or unmistakably non-medical-realism artefacts). Soft stylistic complaints like
    # weak_biology / glassy_cgi_texture / operative_photo_bias no longer force a retry.
    # Very relaxed QA: only regenerate when the image is genuinely unusable
    # (text/UI burned in, wrong subject, or split-screen collage). Stylistic
    # complaints about "feel" no longer trigger retries — good-enough wins.
    always_bad = {"visible_text", "overlay_ui", "collage_or_split_screen", "wrong_intent"}
    biology_bad: set[str] = set()
    concept_bad: set[str] = set()

    if issue_set & always_bad:
        return True
    if seg.intent in BIOLOGY_INTENTS and issue_set & biology_bad:
        return True
    if seg.intent in {"clinical_concept", "data_or_concept"} and issue_set & concept_bad:
        return True
    return False


def _build_regeneration_prompt(prompt: str, seg: Segment, issues: list[str]) -> str:
    """Append corrective guidance for a regeneration attempt."""
    fixes: list[str] = []
    issue_set = set(issues)
    if "visible_text" in issue_set or "overlay_ui" in issue_set:
        fixes.append("Regenerate with absolutely no text, numbers, labels, interface overlays, or monitor typography visible anywhere in frame.")
    if "transparent_mannequin" in issue_set or "plastic_teaching_model" in issue_set:
        fixes.append("Regenerate as organic in-body medical realism, not a plastic teaching model, transparent mannequin, demo torso, or museum specimen.")
    if "symbolic_metaphor" in issue_set:
        fixes.append("Regenerate as a literal medical scene or anatomy frame, not a symbolic metaphor.")
    if "floating_isolated_organ" in issue_set:
        fixes.append("Keep the anatomy embedded in the correct body region with ribs, pleura, vessels, intercostal tissues, and surrounding context visible.")
    if "weak_biology" in issue_set:
        if seg.intent == "molecular":
            fixes.append("Push the biology harder at micro scale: show a specific biochemical action in context such as membrane diffusion, receptor binding, ion-channel flow, vesicle release, hemoglobin gas exchange, or transporter activity with nearby fluid and tissue microenvironment visible.")
        else:
            fixes.append("Push the biology harder: wet pleural membranes, elastic recoil, compression, pressure gradients, thoracic context, and visceral documentary realism.")
    if "glassy_cgi_texture" in issue_set:
        fixes.append("Make the tissues matte, moist, fibrous, and organic. Avoid acrylic translucence, glassy membranes, hollow-shell CGI surfaces, or showroom gloss.")
    if "operative_photo_bias" in issue_set:
        fixes.append("Keep the image educational and anatomically legible rather than resembling an intraoperative gore photo or specimen-table shot.")
    if "wrong_intent" in issue_set and seg.intent in BIOLOGY_INTENTS:
        if seg.intent == "molecular":
            fixes.append("The image must read as a microscopic biochemical frame, not a broad organ exterior, patient exterior shot, or generic concept art.")
        else:
            fixes.append("The image must read as a non-human biology frame, not a patient exterior shot.")

    if not fixes:
        fixes.append("Regenerate a stronger hyperreal medical frame that matches the intended shot type.")

    return prompt + "\n\nREGENERATION FIX:\n" + " ".join(fixes)


def _archive_rejected_attempt(
    rejected_dir: Path,
    seg: Segment,
    attempt: int,
    image_bytes: bytes,
    review: dict[str, object],
    prompt: str,
) -> dict[str, str]:
    """Persist a QA-rejected image attempt for later manual review."""
    rejected_dir.mkdir(parents=True, exist_ok=True)

    issue_tag = _issue_tag(review.get("issues", []))
    stem = f"scene{seg.scene_number}_seg{seg.segment_index}_attempt{attempt}_{issue_tag}"
    image_path = rejected_dir / f"{stem}.png"
    meta_path = rejected_dir / f"{stem}.json"

    image_path.write_bytes(image_bytes)
    meta_path.write_text(json.dumps({
        "scene_number": seg.scene_number,
        "segment_index": seg.segment_index,
        "segment_title": seg.segment_title,
        "intent": seg.intent,
        "attempt": attempt,
        "issues": review.get("issues", []),
        "summary": review.get("summary", ""),
        "prompt": prompt,
        "narration_chunk": seg.narration_chunk,
    }, indent=2))

    return {
        "rejected_image_path": str(image_path),
        "rejected_meta_path": str(meta_path),
    }


def _issue_tag(raw_issues: object) -> str:
    """Build a filesystem-safe short label for archived rejected attempts."""
    if not isinstance(raw_issues, list) or not raw_issues:
        return "unspecified"

    parts: list[str] = []
    for issue in raw_issues[:3]:
        token = str(issue or "").strip().lower().replace(" ", "_").replace("-", "_")
        token = "".join(ch for ch in token if ch.isalnum() or ch == "_").strip("_")
        if token:
            parts.append(token)
    return "_".join(parts) or "unspecified"


def _expected_anatomy_context(seg: Segment, requested_prompt: str = "") -> str:
    """Describe the surrounding anatomy that should usually be visible for certain biology beats."""
    text = f"{seg.segment_title} {seg.narration_chunk} {requested_prompt}".lower()
    if any(keyword in text for keyword in {"lung", "pulmonary", "pleura", "pleural", "pneumothorax", "thoracic", "chest wall", "alveoli"}):
        return (
            "Thoracic anatomy should stay inside a believable chest cavity with ribs, intercostal tissues, pleural lining, and mediastinal context. "
            "If the frame shows a broad bilateral thoracic cross-section, the heart or mediastinum should usually be visible unless the prompt explicitly calls for a tight crop."
        )
    if any(keyword in text for keyword in {"brain", "cortex", "mening", "ventricle", "stroke", "intracranial"}):
        return (
            "Neuroanatomy should preserve skull, meninges, ventricles, or surrounding brain structures so the viewer can orient the lesion or process in the correct region."
        )
    if any(keyword in text for keyword in {"abdomen", "hepatic", "liver", "spleen", "bowel", "intestinal", "peritone"}):
        return (
            "Abdominal anatomy should include surrounding organs, peritoneal boundaries, and body-wall context rather than isolating a single organ in empty space."
        )
    return ""


def _expected_molecular_context(seg: Segment, requested_prompt: str = "") -> str:
    """Describe biologically plausible microenvironments for biochemical process shots."""
    text = f"{seg.segment_title} {seg.narration_chunk} {requested_prompt}".lower()
    if any(keyword in text for keyword in {"oxygen", "o2", "carbon dioxide", "co2", "hemoglobin", "surfactant", "alveolar", "capillary"}):
        return (
            "Show the molecules inside a believable alveolar-capillary microenvironment with thin wet membranes, surfactant, plasma, red blood cells, and capillary endothelium rather than abstract gas particles in empty space."
        )
    if any(keyword in text for keyword in {"synapse", "synaptic", "neurotransmitter", "vesicle", "receptor"}):
        return (
            "Keep the process inside a believable synaptic or receptor-level environment with membranes, vesicles, docking proteins, cleft fluid, and nearby cellular structures visible."
        )
    if any(keyword in text for keyword in {"mitochond", "atp", "proton", "electron transport"}):
        return (
            "Show the process inside a believable organelle microenvironment with membrane folds, protein complexes, fluid compartments, and energetic gradients visible rather than generic glowing particles."
        )
    if any(keyword in text for keyword in {"channel", "sodium", "potassium", "calcium", "chloride", "pump", "transporter", "osmosis"}):
        return (
            "Keep the frame anchored to a real membrane surface with channels, pumps, nearby ions, intracellular and extracellular fluid, and directional transport visible."
        )
    if any(keyword in text for keyword in MOLECULAR_CONTEXT_KEYWORDS):
        return (
            "Keep the microscopic process embedded in a real biological microenvironment with membranes, proteins, fluid, and neighboring structures visible so it feels medical rather than abstract."
        )
    return ""


def _normalize_review_result(seg: Segment, data: dict[str, object]) -> dict[str, object]:
    """Canonicalize QA issues and ensure rejected images always carry a concrete reason."""
    approved = bool(data.get("approved", True))
    summary = str(data.get("summary", "") or "")
    issues = _canonicalize_issue_list(data.get("issues", []))

    for inferred in _infer_issues_from_summary(summary):
        if inferred not in issues:
            issues.append(inferred)

    requires_regen = bool(issues) and _requires_regeneration(seg, issues)

    if requires_regen:
        approved = False
    elif not approved and issues:
        approved = True

    if not approved and not issues:
        approved = True

    return {
        "approved": approved,
        "issues": issues,
        "summary": summary,
    }


def _canonicalize_issue_list(raw_issues: object) -> list[str]:
    """Map model-returned issue names and synonyms onto the supported issue vocabulary."""
    if not isinstance(raw_issues, list):
        return []

    normalized: list[str] = []
    for issue in raw_issues:
        lowered = str(issue or "").strip().lower().replace("-", "_").replace(" ", "_")
        mapped = {
            "text": "visible_text",
            "labels": "visible_text",
            "annotation": "visible_text",
            "annotations": "visible_text",
            "ui": "overlay_ui",
            "plastic_model": "plastic_teaching_model",
            "teaching_model": "plastic_teaching_model",
            "teaching_prop": "plastic_teaching_model",
            "transparent_model": "transparent_mannequin",
            "glassy_texture": "glassy_cgi_texture",
            "cgi_texture": "glassy_cgi_texture",
            "surgical_photo_bias": "operative_photo_bias",
            "operative_photo": "operative_photo_bias",
        }.get(lowered, lowered)
        if mapped in KNOWN_QA_ISSUES and mapped not in normalized:
            normalized.append(mapped)
    return normalized


def _infer_issues_from_summary(summary: str) -> list[str]:
    """Infer canonical QA issue labels from a natural-language review summary."""
    text = (summary or "").lower()
    inferred: list[str] = []

    def add(issue: str) -> None:
        if issue in KNOWN_QA_ISSUES and issue not in inferred:
            inferred.append(issue)

    if any(phrase in text for phrase in {"visible text", "contains text", "text on ", "labels visible", "contains visible blue lines"}):
        add("visible_text")
    if any(phrase in text for phrase in {"overlay ui", "interface overlay", "monitor overlay", "dashboard overlay", "on-screen interface"}):
        add("overlay_ui")
    if "transparent mannequin" in text:
        add("transparent_mannequin")
    if any(phrase in text for phrase in {"plastic teaching model", "plastic model", "teaching prop", "model-like", "museum specimen", "demo torso"}):
        add("plastic_teaching_model")
    if any(phrase in text for phrase in {"symbolic", "metaphor"}):
        add("symbolic_metaphor")
    if any(phrase in text for phrase in {"floating organ", "isolated organ", "floating isolated"}):
        add("floating_isolated_organ")
    if any(phrase in text for phrase in {"glassy", "acrylic", "overly glossy", "hollow-shell", "synthetic cgi", "glass-like", "too glossy"}):
        add("glassy_cgi_texture")
    if any(phrase in text for phrase in {"operative photo", "intraoperative", "surgical view", "specimen shot", "too gory"}):
        add("operative_photo_bias")
    if any(phrase in text for phrase in {"wrong intent", "intent mismatch", "inappropriate for the intended", "rather than a clinically grounded concept", "rather than a clear mechanism illustration", "scene rather than concept"}):
        add("wrong_intent")
    if any(phrase in text for phrase in {"weak biology", "lacking accurate in-body medical realism", "not in-body medical realism", "does not fully satisfy", "overly dramatized", "not a clear mechanism illustration", "limit its educational clarity"}):
        add("weak_biology")
    if any(phrase in text for phrase in {"generic floating molecules", "abstract chemistry art", "molecules in empty space", "not enough microenvironment"}):
        add("weak_biology")
    if "collage" in text or "split-screen" in text or "multi-panel" in text:
        add("collage_or_split_screen")

    return inferred


def engineer_prompts(cues: list[VisualCue]) -> list[ImagePrompt]:
    """Use GPT to engineer hyperreal image prompts from raw visual cues."""
    system_prompt = (PROMPTS_DIR / "image_prompt.txt").read_text()
    prompts: list[ImagePrompt] = []

    for cue in cues:
        user_msg = f"""Transform this visual description into a hyperrealistic cinematic medical image prompt.

**Scene {cue.scene_number}:** {cue.scene_title}
**Visual mode:** {cue.mode}

**Raw visual description:**
{cue.raw_description}

**Surrounding narration:**
{cue.surrounding_narration}

Generate a 16K hyperrealistic cinematic rendering prompt. Think thebrainmaze/thedoctorasky aesthetic — NOT a flat diagram."""

        data = chat_json(
            system_prompt,
            user_msg,
            model=DEFAULT_MODEL,
            max_tokens=1024,
            temperature=0.3,
        )

        prompts.append(ImagePrompt(
            cue=cue,
            prompt=data.get("prompt", cue.raw_description),
            negative_prompt=data.get("negative_prompt", ""),
            style_tags=data.get("style_tags", []),
        ))

    return prompts


def _generate_with_gemini(
    prompt: str,
    reference_images: list[bytes] | None = None,
) -> bytes:
    """Call the Gemini image API with retry on transient network failures.

    Routes based on GEMINI_IMAGE_MODEL:
      - `gemini-*-image-*` (e.g. gemini-3-pro-image-preview / Nano Banana Pro):
        uses :generateContent with inline_data parts. Supports reference images
        for character consistency.
      - `imagen-*` (legacy): uses :predict. Reference images ignored.

    Retries on connection resets, read timeouts, and connect errors with
    exponential backoff. A single flake no longer silently kills an image.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in environment")

    model = GEMINI_IMAGE_MODEL

    last_err: Exception | None = None
    for attempt in range(1, GEMINI_NETWORK_RETRIES + 1):
        try:
            if model.startswith("imagen"):
                if reference_images and attempt == 1:
                    print("      ⚠️  reference_images ignored (Imagen endpoint does not support them)")
                return _imagen_predict(prompt, api_key, model)
            return _gemini_generate_content(prompt, api_key, model, reference_images or [])
        except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError,
                httpx.RemoteProtocolError, httpx.ReadError, OSError) as e:
            last_err = e
            if attempt == GEMINI_NETWORK_RETRIES:
                raise RuntimeError(
                    f"Gemini call failed after {GEMINI_NETWORK_RETRIES} retries: {e}"
                ) from e
            sleep_for = GEMINI_NETWORK_BACKOFF * (2 ** (attempt - 1))
            print(f"      ↻ Gemini retry {attempt}/{GEMINI_NETWORK_RETRIES} "
                  f"in {sleep_for:.0f}s ({type(e).__name__}: {e})")
            time.sleep(sleep_for)

    raise RuntimeError(f"Gemini call exhausted retries: {last_err}")


def _imagen_predict(prompt: str, api_key: str, model: str) -> bytes:
    """Legacy Imagen :predict endpoint — text-only."""
    url = f"{GEMINI_API_BASE}/models/{model}:predict"
    payload = {
        "instances": [{"prompt": prompt}],
        "parameters": {"sampleCount": 1, "aspectRatio": "16:9"},
    }
    response = httpx.post(url, params={"key": api_key}, json=payload, timeout=120.0)
    if response.status_code != 200:
        raise RuntimeError(f"Gemini API error {response.status_code}: {response.text[:500]}")
    data = response.json()
    predictions = data.get("predictions", [])
    if not predictions:
        raise RuntimeError("Gemini returned no predictions")
    image_b64 = predictions[0].get("bytesBase64Encoded", "")
    if not image_b64:
        raise RuntimeError("Gemini returned empty image data")
    return base64.b64decode(image_b64)


def _gemini_generate_content(
    prompt: str,
    api_key: str,
    model: str,
    reference_images: list[bytes],
) -> bytes:
    """Nano Banana Pro :generateContent endpoint with optional reference images."""
    url = f"{GEMINI_API_BASE}/models/{model}:generateContent"

    parts: list[dict] = [{"text": prompt}]
    for ref in reference_images:
        parts.append({
            "inline_data": {
                "mime_type": "image/png",
                "data": base64.b64encode(ref).decode("ascii"),
            }
        })

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
            "imageConfig": {"aspectRatio": "16:9", "imageSize": "2K"},
        },
    }

    response = httpx.post(url, params={"key": api_key}, json=payload, timeout=180.0)
    if response.status_code != 200:
        raise RuntimeError(f"Gemini API error {response.status_code}: {response.text[:500]}")

    data = response.json()
    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {str(data)[:300]}")

    for part in candidates[0].get("content", {}).get("parts", []):
        inline = part.get("inline_data") or part.get("inlineData")
        if inline and inline.get("data"):
            return base64.b64decode(inline["data"])

    raise RuntimeError(f"Gemini response contained no image part: {str(data)[:300]}")
