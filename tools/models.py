"""Pydantic data models for the medical education video pipeline."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


# --- PDF Extraction Models ---


class AnswerChoice(BaseModel):
    letter: str
    text: str


class WrongAnswerExplanation(BaseModel):
    letter: str
    text: str
    explanation: str


class MedicalContent(BaseModel):
    """Structured content extracted from a medical education PDF."""

    topic: str
    subject: str = ""
    system: str = ""
    clinical_vignette: str = ""
    question_stem: str = ""
    answer_choices: list[AnswerChoice] = []
    correct_answer: str = ""
    correct_answer_letter: str = ""
    diagram_description: str = ""
    diagram_labels: list[str] = []
    bottom_line: str = ""
    pathophysiology: str = ""
    key_info: str = ""
    why_section: str = ""
    explanation: str = ""
    wrong_answer_explanations: list[WrongAnswerExplanation] = []
    educational_objective: str = ""


# --- Teaching Plan Models ---


class SceneBrief(BaseModel):
    """High-level outline for a single scene before full script generation."""

    scene_number: int
    scene_title: str
    purpose: str  # hook, question, mechanism, differential, takeaway
    key_content: str
    estimated_minutes: float
    visual_mode: str = "mixed"  # avatar_dominant, animation_dominant, mixed


class DurationOption(BaseModel):
    """A duration recommendation with reasoning."""

    label: str  # "recommended", "minimum", "deep_dive"
    minutes: float
    scene_count: int
    description: str


class TeachingPlan(BaseModel):
    """Structural skeleton that constrains script generation."""

    topic: str
    complexity_score: int = Field(ge=1, le=5)
    concept_count: int
    differential_count: int
    recommended_minutes: float
    duration_options: list[DurationOption] = []
    narrative_hook: str = ""
    tension_point: str = ""
    core_concepts: list[str] = []
    differential_concepts: list[str] = []
    clinical_pearl: str = ""
    scenes: list[SceneBrief] = []


# --- Production Script Models ---


class ProductionScene(BaseModel):
    """A single scene ready for the pipeline."""

    scene: str  # "1 - The Patient Who Couldn't Breathe"
    duration_minutes: float
    word_count: int
    script: str  # Clean narration for TTS (tags stripped)
    script_full: str  # Full narration with [MODE:], [VISUAL:], etc.
    speech_prompt: str = ""  # Avatar delivery cues
    visual_summary: str = ""  # Description of dominant visuals


class ProductionScript(BaseModel):
    """The complete script output for a video."""

    project_name: str
    topic: str
    run_id: str = ""
    pipeline_mode: str = "creative"
    source_kind: str = ""
    grounded: bool = True
    created_at: datetime | None = None
    total_minutes: float
    total_word_count: int
    speech_prompt: str = ""  # Global avatar delivery prompt
    scenes: list[ProductionScene]
    source_pdf: str = ""
    generation_model: str = ""


# --- Phase 2: Image & Voice Models ---


class VisualCue(BaseModel):
    """A single [VISUAL:] tag extracted from a scene's script."""

    scene_number: int
    scene_title: str
    cue_index: int  # Position within the scene (0-based)
    raw_description: str  # Original text from [VISUAL: ...]
    mode: str = ""  # The [MODE:] active when this visual appears
    surrounding_narration: str = ""  # Narration context around the cue


class ImagePrompt(BaseModel):
    """An engineered image generation prompt derived from a VisualCue."""

    cue: VisualCue
    prompt: str  # Optimized prompt for Gemini Imagen
    negative_prompt: str = ""  # What to avoid
    style_tags: list[str] = []  # e.g. ["medical illustration", "cross-section"]


class GeneratedImage(BaseModel):
    """A generated image and its metadata."""

    prompt: ImagePrompt
    file_path: str  # Local path to the saved image
    width: int = 1024
    height: int = 1024


class GeneratedVoice(BaseModel):
    """Generated TTS audio for a scene."""

    scene_number: int
    scene_title: str
    file_path: str  # Local path to the saved audio
    duration_seconds: float = 0.0
    voice_id: str = ""
    model_id: str = ""


# --- Character Consistency Models ---


class CharacterSpec(BaseModel):
    """Canonical description of the patient character used as a reference across segments."""

    age: str = ""
    sex: str = ""
    ethnicity: str = ""
    skin_tone: str = ""
    build: str = ""
    hair: str = ""
    facial_features: str = ""
    accessories: str = ""
    wardrobe: str = ""
    demeanor: str = ""
    continuity_notes: str = ""  # stable identity constraints reused across prompts
    one_line: str = ""  # single-sentence canonical description injected into prompts
    image_path: str = ""  # local path to the generated character reference sheet


# --- Phase 3: Avatar & Animation Models ---


class Segment(BaseModel):
    """A clip-length visual segment within a scene (one Kling clip = one beat)."""

    scene_number: int
    segment_index: int
    segment_title: str  # "Segment 1 - Visual Title"
    image_prompt: str  # Hyperreal image generation prompt
    video_prompt: str  # Motion/animation prompt for image-to-video
    narration_chunk: str  # The narration this segment covers
    duration_seconds: float = 5.0
    start_seconds: float = 0.0
    end_seconds: float = 0.0
    word_count: int = 0
    intent: str = ""  # clinical_scene | patient_experience | exam_or_imaging | mechanism | anatomy | molecular | comparison | clinical_concept | mechanism_summary | data_or_concept(legacy)


class GeneratedAvatar(BaseModel):
    """Generated talking-head avatar video for a scene."""

    scene_number: int
    scene_title: str
    file_path: str
    audio_url: str = ""  # Source audio used
    video_url: str = ""  # Remote URL from Wavespeed


class GeneratedAnimation(BaseModel):
    """Generated medical animation video from a key frame image."""

    scene_number: int
    cue_index: int
    file_path: str
    source_image: str = ""  # Path to the key frame image
    video_url: str = ""  # Remote URL from KIE.ai
    prompt: str = ""  # Motion prompt used


class EvidenceSection(BaseModel):
    """A source section available for grounding and QA."""

    field_name: str
    label: str
    grounded: bool = True
    content: str = ""


class SceneEvidence(BaseModel):
    """Grounding bundle for an individual scene."""

    scene_number: int
    scene_title: str
    purpose: str
    source_sections: list[EvidenceSection] = []


class ReviewIssue(BaseModel):
    """A single review issue found during factual or production QA."""

    severity: str = "warning"  # blocker | warning | note
    category: str = ""
    message: str
    source_sections: list[str] = []


class SceneReview(BaseModel):
    """Scene-level review result."""

    scene: str
    approved: bool = True
    supported: bool = True
    needs_human_review: bool = False
    summary: str = ""
    issues: list[ReviewIssue] = []


class ScriptReview(BaseModel):
    """Full review result for a generated script."""

    approved: bool = True
    grounded: bool = True
    requires_human_review: bool = False
    summary: str = ""
    blockers: list[str] = []
    warnings: list[str] = []
    scenes: list[SceneReview] = []


class RunManifest(BaseModel):
    """Persistent metadata for a single pipeline run."""

    project_name: str
    run_id: str
    pipeline_mode: str
    source_kind: str
    grounded: bool
    source_label: str
    target_minutes: float
    created_at: datetime
    script_review_approved: bool = False
    script_review_summary: str = ""
