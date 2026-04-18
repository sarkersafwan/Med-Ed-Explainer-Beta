"""Compose final video using Remotion for polished output.

Generates a props JSON from pipeline assets, then calls Remotion CLI
to render the final video with:
- Scene transitions (fade)
- Avatar PIP with rounded corners and shadow
- Animated text overlays from [TEXT:] tags
- Audio track synced to video

Falls back to ffmpeg-only composition if Remotion is unavailable.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from tools.models import ProductionScript, Segment

REMOTION_DIR = Path(__file__).parent.parent / "remotion"
FPS = 30


def compose_with_remotion(
    script: ProductionScript,
    segments: list[Segment],
    output_dir: Path,
) -> Path:
    """Build Remotion props and render the final video.

    Returns path to final_video.mp4.
    """
    final_path = output_dir / "final_video.mp4"
    voice_dir = output_dir / "voice"
    avatar_dir = output_dir / "avatars"
    anim_dir = output_dir / "animations"
    images_dir = output_dir / "images"

    # Single source of truth: per-scene mp3 durations drive the timeline.
    # No global concat audio — each scene plays its own mp3 at its own start
    # frame so video + audio can never drift across scenes.
    if not voice_dir.exists():
        print("    ⚠️  No voice dir — cannot compose")
        return final_path

    # Build scene props
    scenes_props = []
    cumulative_frames = 0

    for scene in script.scenes:
        scene_num = _parse_scene_number(scene.scene)
        scene_voice = voice_dir / f"scene{scene_num}.mp3"
        scene_duration_secs = _get_duration(scene_voice) if scene_voice.exists() else scene.duration_minutes * 60
        scene_frames = max(1, int(round(scene_duration_secs * FPS)))

        # Detect mode from script tags. Avatar-mode scenes go full-screen
        # talking head; animation/overlay scenes use the Kling clips.
        scene_mode = "animation"
        if "[MODE: avatar]" in scene.script_full:
            scene_mode = "avatar"
        elif "[MODE: overlay]" in scene.script_full:
            scene_mode = "overlay"

        # Avatar
        avatar_path = avatar_dir / f"scene{scene_num}_avatar.mp4"
        avatar_file = str(avatar_path.resolve()) if avatar_path.exists() else None

        # Segments for this scene (skipped for avatar-mode scenes)
        scene_segments = [
            s for s in segments
            if s.scene_number == scene_num and scene_mode != "avatar"
        ]
        seg_props = []

        if scene_segments:
            # Resize segment slots to exactly fill the scene voice duration so
            # there's never black space at the end. Distribute proportionally
            # to original nominal durations, with a minimum floor.
            nominal = [max(0.5, s.duration_seconds) for s in scene_segments]
            total_nominal = sum(nominal) or 1.0
            slots = [int(round((n / total_nominal) * scene_frames)) for n in nominal]
            # Fix rounding drift so slots sum to scene_frames exactly.
            drift = scene_frames - sum(slots)
            if slots:
                slots[-1] = max(1, slots[-1] + drift)

            for seg, slot_frames in zip(scene_segments, slots):
                anim = anim_dir / f"scene{seg.scene_number}_seg{seg.segment_index}_anim.mp4"
                img = images_dir / f"scene{seg.scene_number}_seg{seg.segment_index}.png"
                anim_secs = _get_duration(anim) if anim.exists() else 0.0
                seg_props.append({
                    "segmentIndex": seg.segment_index,
                    "title": seg.segment_title,
                    "animationFile": str(anim.resolve()) if anim.exists() else None,
                    "imageFile": str(img.resolve()) if img.exists() else None,
                    "durationFrames": slot_frames,
                    "animationDurationFrames": max(1, int(round(anim_secs * FPS))) if anim_secs > 0 else 0,
                })

        # Text overlays from [TEXT:] tags. Spread across the scene timeline
        # rather than clumping at the start, with each overlay held for ~3.5s
        # (or until the next overlay / scene end).
        text_overlays = []
        text_matches = list(re.finditer(r"\[TEXT:\s*([^\]]+)\]", scene.script_full))
        if text_matches:
            n = len(text_matches)
            for idx, match in enumerate(text_matches):
                text = match.group(1).strip().strip('"').strip("'")
                # Distribute starts evenly through the scene, slightly offset.
                start_ratio = (idx + 0.5) / n
                start_frame = int(start_ratio * scene_frames)
                # Hold until the next overlay starts or 4s, whichever is shorter.
                if idx + 1 < n:
                    next_start = int(((idx + 1.5) / n) * scene_frames)
                    duration_frames = max(int(2 * FPS), next_start - start_frame - int(0.3 * FPS))
                else:
                    duration_frames = max(int(2 * FPS), scene_frames - start_frame - int(0.3 * FPS))
                duration_frames = min(duration_frames, int(5 * FPS))
                text_overlays.append({
                    "text": text,
                    "startFrame": start_frame,
                    "durationFrames": duration_frames,
                    "emphasis": _detect_emphasis(text),
                })

        scenes_props.append({
            "sceneNumber": scene_num,
            "title": scene.scene,
            "mode": scene_mode,
            "durationFrames": scene_frames,
            "voiceFile": str(scene_voice.resolve()) if scene_voice.exists() else None,
            "avatarFile": avatar_file,
            "segments": seg_props,
            "textOverlays": text_overlays,
        })

        cumulative_frames += scene_frames

    # Total frames is the authoritative sum of all scene frames — never
    # computed from a separately concat'd audio file which would round
    # differently and introduce trailing black space.
    total_frames = cumulative_frames

    # Build the full props — no top-level audio; per-scene audio lives in
    # each scene's voiceFile and is played inside the scene sequence.
    props = {
        "scenes": scenes_props,
        "audioFile": "",
        "fps": FPS,
        "totalDurationFrames": total_frames,
    }

    # Copy assets to a run-scoped public dir so one render cannot reuse another run's files
    asset_namespace = script.run_id or output_dir.name
    public_dir = REMOTION_DIR / "public" / "render_assets" / asset_namespace
    public_dir.mkdir(parents=True, exist_ok=True)

    def _copy_to_public(local_path: str | None) -> str | None:
        if not local_path:
            return None
        src = Path(local_path)
        if not src.exists():
            return None
        dest = public_dir / src.name
        shutil.copy2(src, dest)
        return f"render_assets/{asset_namespace}/{src.name}"

    # Update props to use staticFile paths
    for scene in props["scenes"]:
        scene["avatarFile"] = _copy_to_public(scene["avatarFile"])
        scene["voiceFile"] = _copy_to_public(scene.get("voiceFile"))
        for seg in scene["segments"]:
            seg["animationFile"] = _copy_to_public(seg["animationFile"])
            seg["imageFile"] = _copy_to_public(seg["imageFile"])

    # Write props JSON
    props_path = output_dir / "remotion_props.json"
    props_path.write_text(json.dumps(props, indent=2))
    print(f"    Props: {props_path}")

    # Render with Remotion CLI
    print(f"    Rendering with Remotion ({total_frames} frames, {total_frames / FPS:.1f}s)...")
    try:
        result = subprocess.run(
            [
                "npx", "remotion", "render",
                "src/index.ts",
                "MedicalVideo",
                str(final_path.resolve()),
                "--props", str(props_path.resolve()),
                "--codec", "h264",
                "--concurrency", "4",
            ],
            cwd=str(REMOTION_DIR),
            capture_output=True,
            text=True,
            timeout=900,
        )

        if result.returncode == 0:
            print(f"    ✓ Remotion render complete: {final_path}")
        else:
            print(f"    ⚠️  Remotion render failed, falling back to ffmpeg")
            print(f"    Error: {result.stderr[:300]}")
            # Fall back to ffmpeg composition
            from tools.compose import compose_video
            return compose_video(script, segments, output_dir)

    except FileNotFoundError:
        print("    ⚠️  Remotion not installed, falling back to ffmpeg")
        from tools.compose import compose_video
        return compose_video(script, segments, output_dir)
    except subprocess.TimeoutExpired:
        print("    ⚠️  Remotion render timed out, falling back to ffmpeg")
        from tools.compose import compose_video
        return compose_video(script, segments, output_dir)

    return final_path


def _concat_audio(voice_dir: Path, output_dir: Path) -> Path | None:
    """Concatenate scene MP3s into one audio file."""
    if not voice_dir.exists():
        return None
    mp3s = sorted(voice_dir.glob("scene*.mp3"), key=_scene_media_sort_key)
    if not mp3s:
        return None
    if len(mp3s) == 1:
        return mp3s[0]

    concat_list = output_dir / "_audio_list.txt"
    with open(concat_list, "w") as f:
        for mp3 in mp3s:
            f.write(f"file '{mp3.resolve()}'\n")

    output = output_dir / "_full_audio.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "concat", "-safe", "0", "-i", str(concat_list),
         "-c", "copy", str(output)],
        capture_output=True,
    )
    concat_list.unlink(missing_ok=True)
    return output if output.exists() else None


def _get_duration(path: Path) -> float:
    """Get media duration in seconds."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True,
        )
        return float(result.stdout.strip())
    except (ValueError, FileNotFoundError):
        return 0.0


def _detect_emphasis(text: str) -> str:
    """Classify a [TEXT:] overlay so the front-end can style it appropriately.

    Clinical-knowledge overlays (definitions, mechanisms, key takeaways) deserve
    bigger, longer, higher-contrast treatment than throwaway labels.
    """
    lower = text.lower()
    clinical_keywords = (
        "→", "->", "=", "blocks", "inhibits", "causes", "leads to",
        "binds", "stops", "prevents", "mechanism", "pathway",
        "oxygen", "atp", "hypoxia", "respiration", "receptor",
    )
    if any(k in lower for k in clinical_keywords) or len(text) > 30:
        return "clinical"
    return "label"


def _parse_scene_number(scene_label: str) -> int:
    match = re.match(r"(\d+)", scene_label.strip())
    return int(match.group(1)) if match else 0


def _scene_media_sort_key(path: Path) -> tuple[int, str]:
    """Sort scene media numerically so scene10 comes after scene9."""
    match = re.search(r"scene(\d+)", path.name)
    return (int(match.group(1)) if match else 0, path.name)
