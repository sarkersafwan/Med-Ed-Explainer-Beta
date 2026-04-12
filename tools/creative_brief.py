"""Parse and structure a creative brief for video generation.

The creative brief is the full INPUT_Request from Airtable — it contains
topic, duration, voice direction, avatar expression cues, visual style
preferences, and text overlay choices.

Also handles style reference image analysis (INPUT_Image_2).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import httpx

from tools.provider import chat_json, get_text_model_name, vision_text

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", get_text_model_name())


class CreativeBrief:
    """Structured creative direction for the entire video pipeline."""

    def __init__(
        self,
        topic: str = "",
        duration_minutes: float = 0,
        voice_direction: str = "",
        avatar_expression: str = "",
        visual_style: str = "",
        color_palette: str = "",
        text_overlay_style: str = "",
        full_request: str = "",
        style_reference_url: str = "",
        style_analysis: str = "",
    ):
        self.topic = topic
        self.duration_minutes = duration_minutes
        self.voice_direction = voice_direction
        self.avatar_expression = avatar_expression
        self.visual_style = visual_style
        self.color_palette = color_palette
        self.text_overlay_style = text_overlay_style
        self.full_request = full_request
        self.style_reference_url = style_reference_url
        self.style_analysis = style_analysis

    def to_script_prompt(self) -> str:
        """Generate the creative direction section for script generation."""
        parts = []
        if self.voice_direction:
            parts.append(f"VOICE DIRECTION: {self.voice_direction}")
        if self.avatar_expression:
            parts.append(f"AVATAR EXPRESSION: {self.avatar_expression}")
        if self.visual_style:
            parts.append(f"VISUAL STYLE: {self.visual_style}")
        return "\n".join(parts) if parts else ""

    def to_image_style_prompt(self) -> str:
        """Generate style direction for image/segment generation."""
        parts = []
        if self.visual_style:
            parts.append(f"Visual style: {self.visual_style}")
        if self.color_palette:
            parts.append(f"Color palette: {self.color_palette}")
        if self.style_analysis:
            parts.append(f"Style reference analysis: {self.style_analysis}")
        return "\n".join(parts) if parts else "Hyperrealistic cinematic medical rendering, dark background, volumetric lighting"

    def to_airtable_request(self) -> str:
        """Generate the full INPUT_Request for Airtable."""
        if self.full_request:
            return self.full_request

        parts = [f"Topic: {self.topic}"]
        if self.duration_minutes:
            parts.append(f"Duration: {self.duration_minutes} minutes")
        if self.voice_direction:
            parts.append(f"\nVoice: {self.voice_direction}")
        if self.avatar_expression:
            parts.append(f"Avatar: {self.avatar_expression}")
        if self.visual_style:
            parts.append(f"Visual style: {self.visual_style}")
        if self.color_palette:
            parts.append(f"Colors: {self.color_palette}")
        if self.text_overlay_style:
            parts.append(f"Text overlay: {self.text_overlay_style}")
        return "\n".join(parts)

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "duration_minutes": self.duration_minutes,
            "voice_direction": self.voice_direction,
            "avatar_expression": self.avatar_expression,
            "visual_style": self.visual_style,
            "color_palette": self.color_palette,
            "text_overlay_style": self.text_overlay_style,
            "style_analysis": self.style_analysis,
            "full_request": self.full_request,
        }


def parse_creative_brief(request_text: str, duration: float = 0) -> CreativeBrief:
    """Parse a free-form creative brief into structured fields using GPT."""
    if not request_text.strip():
        return CreativeBrief()

    data = chat_json(
        """You parse creative briefs for medical education video production.
Extract structured fields from the user's request. Return JSON:
{
  "topic": "the medical topic to teach",
  "duration_minutes": 0,
  "voice_direction": "tone, style, pacing instructions for TTS",
  "avatar_expression": "facial expression and body language cues",
  "visual_style": "b-roll/animation visual style description",
  "color_palette": "color preferences (hues, mode, contrast)",
  "text_overlay_style": "font, color, positioning for text overlays"
}
If a field isn't mentioned, use empty string. Return ONLY JSON.""",
        request_text,
        model=DEFAULT_MODEL,
        max_tokens=1024,
        temperature=0.1,
    )

    return CreativeBrief(
        topic=data.get("topic", ""),
        duration_minutes=data.get("duration_minutes", 0) or duration,
        voice_direction=data.get("voice_direction", ""),
        avatar_expression=data.get("avatar_expression", ""),
        visual_style=data.get("visual_style", ""),
        color_palette=data.get("color_palette", ""),
        text_overlay_style=data.get("text_overlay_style", ""),
        full_request=request_text,
    )


def analyze_style_reference(image_path: str) -> str:
    """Analyze a style reference image to extract visual style keywords.

    Uses GPT-4 Vision to describe the rendering style, lighting, colors,
    and overall aesthetic — which then gets baked into image generation prompts.
    """
    # Read and encode the image
    if image_path.startswith("http"):
        r = httpx.get(image_path, follow_redirects=True, timeout=30)
        image_bytes = r.content
        mime = "image/jpeg"
    else:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        ext = Path(image_path).suffix.lower()
        mime = {"png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(ext, "image/jpeg")

    return vision_text(
        """You analyze medical illustration reference images for style extraction.
Describe the visual style in terms that can be used as image generation prompt modifiers.
Focus on: rendering style, lighting type, color palette, background treatment, level of realism,
material properties (translucency, glow, texture), camera angle, and overall mood.
Output a concise style description (3-5 sentences) that can be prepended to image generation prompts.""",
        "Analyze this medical illustration style reference. Describe its visual style for image generation:",
        image_bytes=image_bytes,
        mime_type=mime,
        model=DEFAULT_MODEL,
        max_tokens=512,
    )


def build_brief_from_inputs(
    topic: str = "",
    request_text: str = "",
    duration: float = 0,
    style_image_path: str = "",
) -> CreativeBrief:
    """Build a complete creative brief from available inputs.

    Args:
        topic: Simple topic string (used if request_text is empty)
        request_text: Full creative brief text (INPUT_Request from Airtable)
        duration: Target duration in minutes
        style_image_path: Path or URL to style reference image (INPUT_Image_2)
    """
    # Parse the creative brief
    if request_text:
        print("   Parsing creative brief...")
        brief = parse_creative_brief(request_text, duration)
    else:
        brief = CreativeBrief(
            topic=topic,
            duration_minutes=duration,
            full_request=f"Topic: {topic}",
        )

    # Analyze style reference if provided
    if style_image_path:
        print("   Analyzing style reference image...")
        brief.style_analysis = analyze_style_reference(style_image_path)
        brief.style_reference_url = style_image_path
        print(f"   Style: {brief.style_analysis[:100]}...")

    return brief
