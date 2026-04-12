from tools.character_sheet import _build_character_sheet_prompt
from tools.generate_images import (
    _archive_rejected_attempt,
    _normalize_review_result,
    _requires_regeneration,
    _tighten_segment_prompt,
)
from tools.generate_segments import _normalize_segment_intent
from tools.models import CharacterSpec, Segment


def test_clinical_concept_prompt_is_grounded_and_single_frame():
    seg = Segment(
        scene_number=1,
        segment_index=0,
        segment_title="Concept",
        image_prompt="Hyperreal concept image of rising pressure.",
        video_prompt="x",
        narration_chunk="Rising pressure.",
        intent="clinical_concept",
    )

    tightened = _tighten_segment_prompt(seg, seg.image_prompt)
    assert "single coherent hyperreal frame" in tightened
    assert "Do not use storm clouds" in tightened
    assert "text-free medical visual" in tightened or "text-free clinical artifact" in tightened
    assert "clipboards" in tightened
    assert "plastic teaching models" in tightened


def test_mechanism_prompt_requires_in_situ_anatomy():
    seg = Segment(
        scene_number=3,
        segment_index=0,
        segment_title="Mechanism",
        image_prompt="Hyperreal heart failure mechanism render.",
        video_prompt="x",
        narration_chunk="Pressure backs up into the lungs.",
        intent="mechanism",
    )

    tightened = _tighten_segment_prompt(seg, seg.image_prompt)
    assert "in situ within the correct body region" in tightened
    assert "No floating isolated organs" in tightened


def test_mechanism_summary_prompt_forces_in_body_teaching():
    seg = Segment(
        scene_number=3,
        segment_index=0,
        segment_title="Mechanism Summary",
        image_prompt="Hyperreal pneumothorax summary render.",
        video_prompt="x",
        narration_chunk="Pleural air breaks the seal and the lung collapses.",
        intent="mechanism_summary",
    )

    tightened = _tighten_segment_prompt(seg, seg.image_prompt)
    assert "visceral in-body anatomy" in tightened
    assert "No plastic teaching model" in tightened
    assert "heart or mediastinum" in tightened
    assert "matte, moist, organic pleura" in tightened


def test_molecular_prompt_requires_micro_biochemical_context():
    seg = Segment(
        scene_number=2,
        segment_index=1,
        segment_title="Oxygen Diffusion",
        image_prompt="Hyperreal oxygen diffusion render.",
        video_prompt="x",
        narration_chunk="Oxygen diffuses across the alveolar membrane and binds hemoglobin.",
        intent="molecular",
    )

    tightened = _tighten_segment_prompt(seg, seg.image_prompt)
    assert "microscopic or biochemical scale" in tightened
    assert "membranes, receptors, channels, vesicles" in tightened
    assert "alveolar-capillary microenvironment" in tightened
    assert "no broad whole-organ overview" in tightened


def test_character_sheet_prompt_requests_multi_angle_sheet():
    spec = CharacterSpec(
        one_line="A 56-year-old male, average build, short dark hair, blue button-down shirt, tired and breathless.",
        skin_tone="light olive",
        facial_features="soft under-eye bags, square jaw, broad nose",
        accessories="none",
        continuity_notes="Keep the same face shape, hairline, and shirt in every view.",
    )

    prompt = _build_character_sheet_prompt(spec)
    assert "six-view contact sheet" in prompt
    assert "front portrait" in prompt
    assert "left profile" in prompt
    assert "full-body standing view" in prompt


def test_legacy_data_or_concept_maps_to_mechanism_summary_in_mechanism_scene():
    intent = _normalize_segment_intent(
        "data_or_concept",
        scene_purpose="mechanism",
        segment_title="Air breaks the seal",
        narration_chunk="Pleural air causes lung collapse.",
    )
    assert intent == "mechanism_summary"


def test_short_explainer_clinical_concept_maps_to_mechanism_summary():
    intent = _normalize_segment_intent(
        "clinical_concept",
        scene_purpose="takeaway",
        segment_title="Clinical reminder",
        narration_chunk="Pleural air breaks the seal and the lung collapses.",
        total_video_minutes=0.5,
    )
    assert intent == "mechanism_summary"


def test_image_qa_rejects_text_and_teaching_models():
    seg = Segment(
        scene_number=3,
        segment_index=1,
        segment_title="Mechanism",
        image_prompt="x",
        video_prompt="x",
        narration_chunk="x",
        intent="mechanism",
    )

    assert _requires_regeneration(seg, ["visible_text"]) is True
    assert _requires_regeneration(seg, ["plastic_teaching_model"]) is True
    assert _requires_regeneration(seg, ["glassy_cgi_texture"]) is True


def test_qa_summary_infers_reject_reason_when_issue_list_is_empty():
    seg = Segment(
        scene_number=2,
        segment_index=0,
        segment_title="Negative Pressure Seal",
        image_prompt="x",
        video_prompt="x",
        narration_chunk="Negative pressure keeps the lung against the chest wall.",
        intent="mechanism",
    )

    review = _normalize_review_result(
        seg,
        {
            "approved": False,
            "issues": [],
            "summary": "The image looks more like a surgical/operative photo than a clear mechanism illustration.",
        },
    )

    assert review["approved"] is False
    assert "operative_photo_bias" in review["issues"]


def test_non_gating_anatomy_note_does_not_force_rejection():
    seg = Segment(
        scene_number=2,
        segment_index=0,
        segment_title="Thoracic overview",
        image_prompt="x",
        video_prompt="x",
        narration_chunk="Pleural air separates the lung from the chest wall.",
        intent="mechanism_summary",
    )

    review = _normalize_review_result(
        seg,
        {
            "approved": False,
            "issues": [],
            "summary": "The anatomy could use more surrounding thoracic context.",
        },
    )

    assert review["approved"] is True


def test_rejected_attempt_is_archived(tmp_path):
    seg = Segment(
        scene_number=3,
        segment_index=2,
        segment_title="Lung Collapse Mechanism",
        image_prompt="x",
        video_prompt="x",
        narration_chunk="Pleural air causes collapse.",
        intent="mechanism_summary",
    )

    result = _archive_rejected_attempt(
        tmp_path / "rejected",
        seg,
        2,
        b"fakepng",
        {
            "approved": False,
            "issues": ["missing_key_anatomy", "glassy_cgi_texture"],
            "summary": "Missing thoracic context and looks too glossy.",
        },
        "Prompt text",
    )

    image_path = tmp_path / "rejected" / "scene3_seg2_attempt2_missing_key_anatomy_glassy_cgi_texture.png"
    meta_path = tmp_path / "rejected" / "scene3_seg2_attempt2_missing_key_anatomy_glassy_cgi_texture.json"

    assert result["rejected_image_path"] == str(image_path)
    assert result["rejected_meta_path"] == str(meta_path)
    assert image_path.exists()
    assert meta_path.exists()
