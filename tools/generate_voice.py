"""Generate TTS voice audio for each scene using ElevenLabs.

Takes the clean narration (tags stripped) from each scene and generates
natural-sounding speech audio. Uses speech_prompt for delivery direction.
"""

from __future__ import annotations

import os
import struct
import wave
from pathlib import Path

from elevenlabs import ElevenLabs

from tools.models import GeneratedVoice, ProductionScript
from tools.parallel import run_parallel, safe_print

# Default voice — can be overridden via env or CLI arg
DEFAULT_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "")
DEFAULT_MODEL = os.environ.get("ELEVENLABS_MODEL", "eleven_multilingual_v2")


def generate_voice(
    script: ProductionScript,
    output_dir: Path,
    voice_id: str = "",
    model_id: str = "",
    skip_existing: bool = True,
) -> list[GeneratedVoice]:
    """Generate TTS audio for every scene in the script.

    Args:
        script: The production script with scenes.
        output_dir: Directory to save audio files.
        voice_id: ElevenLabs voice ID (falls back to env/default).
        model_id: ElevenLabs model ID.
        skip_existing: Skip scenes that already have audio on disk.

    Returns:
        List of GeneratedVoice objects with file paths.
    """
    voice_dir = output_dir / "voice"
    voice_dir.mkdir(parents=True, exist_ok=True)

    vid = voice_id or DEFAULT_VOICE_ID
    mid = model_id or DEFAULT_MODEL

    if not vid:
        raise ValueError(
            "No voice ID provided. Set ELEVENLABS_VOICE_ID in .env or pass --voice-id"
        )

    api_key = os.environ.get("ELEVENLABS_API_KEY", "")
    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY not set in environment")

    client = ElevenLabs(api_key=api_key)

    def _one_voice(scene, _idx: int) -> GeneratedVoice | None:
        scene_num = _parse_scene_number(scene.scene)
        filename = f"scene{scene_num}.mp3"
        filepath = voice_dir / filename

        if skip_existing and filepath.exists():
            safe_print(f"    [scene {scene_num}] skip (exists): {filename}")
            return GeneratedVoice(
                scene_number=scene_num, scene_title=scene.scene,
                file_path=str(filepath), voice_id=vid, model_id=mid,
            )

        text = scene.script
        if not text.strip():
            safe_print(f"    ⚠️  [scene {scene_num}] empty narration, skipping")
            return None

        safe_print(f"    [scene {scene_num}] generating voice…")
        audio_generator = client.text_to_speech.convert(
            voice_id=vid, text=text, model_id=mid,
        )
        audio_bytes = b"".join(audio_generator)
        filepath.write_bytes(audio_bytes)
        safe_print(f"    [scene {scene_num}] ✓ saved {filename} "
                   f"({len(audio_bytes) / 1024:.0f} KB)")

        return GeneratedVoice(
            scene_number=scene_num, scene_title=scene.scene,
            file_path=str(filepath), voice_id=vid, model_id=mid,
        )

    results = run_parallel(
        list(script.scenes),
        _one_voice,
        max_workers=int(os.environ.get("VOICE_PARALLEL", "4")),
        label="voices",
    )
    return [r.value for r in results if r.ok and r.value is not None]


def list_voices() -> list[dict]:
    """List available ElevenLabs voices for selection."""
    api_key = os.environ.get("ELEVENLABS_API_KEY", "")
    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY not set in environment")

    client = ElevenLabs(api_key=api_key)
    response = client.voices.get_all()

    return [
        {"voice_id": v.voice_id, "name": v.name, "category": v.category}
        for v in response.voices
    ]


def _parse_scene_number(scene_label: str) -> int:
    """Extract scene number from labels like '3 - The Mechanism'."""
    import re
    match = re.match(r"(\d+)", scene_label.strip())
    return int(match.group(1)) if match else 0
