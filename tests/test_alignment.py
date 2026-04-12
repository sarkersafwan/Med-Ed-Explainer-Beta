import tools.generate_segments as generate_segments_mod
from tools.alignment import assign_segment_timings, validate_segment_coverage
from tools.generate_segments import (
    _deterministic_narration_chunks,
    _fallback_intent_for_chunk,
    _generate_scene_segments,
    _normalize_segment_intent,
    _validate_scene_visual_balance,
)
from tools.models import ProductionScene, Segment


def test_segment_coverage_and_timing_assignment():
    scene = ProductionScene(
        scene='1 - Test',
        duration_minutes=0.2,
        word_count=0,
        script='Blood backs up into the lungs. Fluid leaks into the alveoli.',
        script_full='[MODE: animation] Blood backs up into the lungs. Fluid leaks into the alveoli.',
    )
    segments = [
        Segment(
            scene_number=1,
            segment_index=0,
            segment_title='Back pressure',
            image_prompt='x',
            video_prompt='x',
            narration_chunk='Blood backs up into the lungs.',
            intent='mechanism',
        ),
        Segment(
            scene_number=1,
            segment_index=1,
            segment_title='Alveolar edema',
            image_prompt='x',
            video_prompt='x',
            narration_chunk='Fluid leaks into the alveoli.',
            intent='mechanism',
        ),
    ]

    assert validate_segment_coverage(scene, segments) == []
    assigned = assign_segment_timings(scene, segments)
    assert assigned[0].start_seconds == 0.0
    assert assigned[-1].end_seconds == 12.0
    assert sum(seg.word_count for seg in assigned) > 0


def test_visual_balance_prefers_biology_in_mechanism_scenes():
    scene = ProductionScene(
        scene='2 - The Mechanism',
        duration_minutes=0.3,
        word_count=0,
        script='Pressure rises and fluid backs up.',
        script_full='[MODE: animation] Pressure rises and fluid backs up.',
    )
    segments = [
        Segment(
            scene_number=2,
            segment_index=0,
            segment_title='Patient reaction',
            image_prompt='x',
            video_prompt='x',
            narration_chunk='Pressure rises.',
            intent='clinical_scene',
        ),
        Segment(
            scene_number=2,
            segment_index=1,
            segment_title='More patient reaction',
            image_prompt='x',
            video_prompt='x',
            narration_chunk='Fluid backs up.',
            intent='patient_experience',
        ),
    ]

    issues = _validate_scene_visual_balance(scene, segments)
    assert issues
    assert any('biology-first visuals' in issue or 'too many human-intent shots' in issue for issue in issues)


def test_hook_scene_allows_some_human_setup_before_biology():
    scene = ProductionScene(
        scene='1 - The Patient',
        duration_minutes=0.8,
        word_count=0,
        script='Patient arrives short of breath before explanation moves into physiology.',
        script_full='[MODE: avatar] Patient arrives short of breath before explanation moves into physiology.',
    )
    segments = [
        Segment(
            scene_number=1,
            segment_index=0,
            segment_title='Arrival',
            image_prompt='x',
            video_prompt='x',
            narration_chunk='Patient arrives.',
            intent='clinical_scene',
        ),
        Segment(
            scene_number=1,
            segment_index=1,
            segment_title='Shortness of breath',
            image_prompt='x',
            video_prompt='x',
            narration_chunk='Short of breath.',
            intent='patient_experience',
        ),
        Segment(
            scene_number=1,
            segment_index=2,
            segment_title='Focused exam detail',
            image_prompt='x',
            video_prompt='x',
            narration_chunk='Focused exam detail.',
            intent='exam_or_imaging',
        ),
        Segment(
            scene_number=1,
            segment_index=3,
            segment_title='CXR',
            image_prompt='x',
            video_prompt='x',
            narration_chunk='Chest x-ray.',
            intent='exam_or_imaging',
        ),
        Segment(
            scene_number=1,
            segment_index=4,
            segment_title='Pulmonary veins',
            image_prompt='x',
            video_prompt='x',
            narration_chunk='Pulmonary veins.',
            intent='mechanism',
        ),
    ]

    issues = _validate_scene_visual_balance(scene, segments)
    assert issues == []


def test_segment_generation_retries_after_invalid_json(monkeypatch):
    scene = ProductionScene(
        scene='1 - The Patient',
        duration_minutes=0.1,
        word_count=0,
        script='He arrives short of breath.',
        script_full='[MODE: avatar] He arrives short of breath.',
    )
    responses = iter([
        'not valid json',
        '[{"segment_title":"Arrival","intent":"clinical_scene","narration_chunk":"He arrives short of breath.","image_prompt":"ED arrival image","video_prompt":"Patient walks in"}]',
    ])

    monkeypatch.setattr(generate_segments_mod, 'chat_text', lambda *args, **kwargs: next(responses))

    segments = _generate_scene_segments(
        'image prompt system',
        'video prompt system',
        scene,
        1,
        1,
        max_retries=1,
    )

    assert len(segments) == 1
    assert segments[0].narration_chunk == 'He arrives short of breath.'


def test_segment_generation_accepts_balance_warning_after_retries(monkeypatch):
    scene = ProductionScene(
        scene='1 - The Patient',
        duration_minutes=0.2,
        word_count=0,
        script='He arrives short of breath. He looks exhausted.',
        script_full='[MODE: avatar] He arrives short of breath. He looks exhausted.',
    )

    monkeypatch.setattr(
        generate_segments_mod,
        'chat_text',
        lambda *args, **kwargs: (
            '['
            '{"segment_title":"Arrival","intent":"clinical_scene","narration_chunk":"He arrives short of breath.","image_prompt":"ED arrival image","video_prompt":"Patient walks in"},'
            '{"segment_title":"Fatigue","intent":"patient_experience","narration_chunk":"He looks exhausted.","image_prompt":"Close-up of fatigue","video_prompt":"Patient sags in frame"}'
            ']'
        ),
    )

    segments = _generate_scene_segments(
        'image prompt system',
        'video prompt system',
        scene,
        1,
        2,
        max_retries=0,
    )

    assert len(segments) == 2
    assert segments[0].start_seconds == 0.0


def test_short_explainer_flags_text_prone_clinical_concepts():
    scene = ProductionScene(
        scene='3 - Takeaway',
        duration_minutes=0.2,
        word_count=0,
        script='Air in the pleural space collapses the lung.',
        script_full='[MODE: animation] Air in the pleural space collapses the lung.',
    )
    segments = [
        Segment(
            scene_number=3,
            segment_index=0,
            segment_title='Clinical reminder',
            image_prompt='Monitor bank in triage bay.',
            video_prompt='x',
            narration_chunk='Air in the pleural space collapses the lung.',
            intent='clinical_concept',
        ),
    ]

    issues = _validate_scene_visual_balance(scene, segments, total_video_minutes=0.5)
    assert any('Short explainers should avoid text-prone clinical_concept shots' in issue for issue in issues)


def test_deterministic_chunking_reconstructs_scene_text():
    narration = (
        "A 25-year-old tall, thin man suddenly gasps, clutching his right chest, struggling to breathe. "
        "Imagine a door slamming shut inside his chest, causing sharp pain and shortness of breath. "
        "His breathing quickens, and you notice he looks anxious and uncomfortable."
    )

    chunks = _deterministic_narration_chunks(narration, 4)

    assert len(chunks) == 4
    assert " ".join(chunk.strip() for chunk in chunks)
    scene = ProductionScene(
        scene='1 - The Patient',
        duration_minutes=0.1,
        word_count=0,
        script=narration,
        script_full=narration,
    )
    segments = [
        Segment(
            scene_number=1,
            segment_index=i,
            segment_title=f'Segment {i+1}',
            image_prompt='x',
            video_prompt='x',
            narration_chunk=chunk,
            intent='mechanism_summary' if i > 1 else 'clinical_scene',
        )
        for i, chunk in enumerate(chunks)
    ]
    assert validate_segment_coverage(scene, segments) == []


def test_molecular_narration_maps_to_molecular_intent():
    intent = _normalize_segment_intent(
        "mechanism_summary",
        scene_purpose="mechanism",
        segment_title="Gas Exchange",
        narration_chunk="Oxygen diffuses across the alveolar membrane and binds hemoglobin in capillary red blood cells.",
        total_video_minutes=0.5,
    )
    assert intent == "molecular"


def test_non_biochemical_pleural_mechanics_do_not_stay_molecular():
    intent = _normalize_segment_intent(
        "molecular",
        scene_purpose="mechanism",
        segment_title="Normal pleural pressure",
        narration_chunk="Your lung stays inflated because negative pressure in the pleural space keeps the visceral and parietal pleura apposed.",
        total_video_minutes=0.5,
    )
    assert intent == "mechanism"


def test_fallback_prefers_molecular_for_biochemical_chunk():
    intent = _fallback_intent_for_chunk(
        "Calcium rushes through ion channels and triggers vesicle release at the synapse.",
        index=1,
        total_chunks=3,
        scene_purpose="mechanism",
        total_video_minutes=0.5,
    )
    assert intent == "molecular"
