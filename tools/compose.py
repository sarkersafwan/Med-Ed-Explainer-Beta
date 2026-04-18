"""Compose final video from generated assets using ffmpeg.

Mode-aware composition based on [MODE:] tags in the script:
- [MODE: avatar] → full-screen avatar talking head
- [MODE: animation] → full-screen segment animation/image
- [MODE: overlay] → avatar PIP in corner over animation background

Audio is the timeline driver — total video length matches total voice duration.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from tools.models import ProductionScript, Segment


def compose_video(
    script: ProductionScript,
    segments: list[Segment],
    output_dir: Path,
    resolution: str = "1920x1080",
    fps: int = 30,
) -> Path:
    """Compose the final video from all generated assets.

    Returns path to final_video.mp4.
    """
    final_path = output_dir / "final_video.mp4"
    voice_dir = output_dir / "voice"
    avatar_dir = output_dir / "avatars"
    anim_dir = output_dir / "animations"
    images_dir = output_dir / "images"

    width, height = [int(x) for x in resolution.split("x")]

    # Step 1: Build master audio track
    print("    Building audio track...")
    audio_path = _concat_audio(voice_dir, output_dir)
    if not audio_path:
        print("    ⚠️  No audio files — cannot compose video")
        return final_path

    # Get audio duration for timing
    audio_duration = _get_duration(audio_path)
    print(f"    Audio duration: {audio_duration:.1f}s")

    # Step 2: Build per-scene video clips
    print("    Building scene clips...")
    scene_clips = []

    for i, scene in enumerate(script.scenes):
        scene_num = _parse_scene_number(scene.scene)
        scene_voice = voice_dir / f"scene{scene_num}.mp3"
        scene_duration = _get_duration(scene_voice) if scene_voice.exists() else scene.duration_minutes * 60

        # Get available assets for this scene
        avatar_path = avatar_dir / f"scene{scene_num}_avatar.mp4"
        scene_segments = [s for s in segments if s.scene_number == scene_num]

        # Build this scene's clip based on available assets
        scene_clip = output_dir / f"_scene{scene_num}_clip.mp4"

        # Get the best visual for this scene
        visual_clips = []
        for seg in scene_segments:
            anim = anim_dir / f"scene{seg.scene_number}_seg{seg.segment_index}_anim.mp4"
            img = images_dir / f"scene{seg.scene_number}_seg{seg.segment_index}.png"
            if anim.exists():
                visual_clips.append(("animation", anim, seg))
            elif img.exists():
                visual_clips.append(("image", img, seg))

        if avatar_path.exists() and visual_clips:
            # We have both avatar and visuals — create overlay composition
            print(f"    Scene {scene_num}: avatar + {len(visual_clips)} visuals (overlay mode)")
            _compose_scene_overlay(
                avatar_path, visual_clips, scene_clip,
                scene_duration, width, height, fps
            )
        elif avatar_path.exists():
            # Avatar only — full screen
            print(f"    Scene {scene_num}: avatar only (full screen)")
            _scale_and_trim(avatar_path, scene_clip, scene_duration, width, height)
        elif visual_clips:
            # Visuals only — sequence them
            print(f"    Scene {scene_num}: {len(visual_clips)} visuals (no avatar)")
            _compose_scene_visuals(visual_clips, scene_clip, scene_duration, width, height, fps)
        else:
            # Nothing — create a dark branded frame (never pure black)
            print(f"    Scene {scene_num}: no assets, branded placeholder")
            _create_branded_clip(scene_clip, scene_duration, width, height, fps)

        if scene_clip.exists():
            scene_clips.append(scene_clip)

    if not scene_clips:
        print("    ⚠️  No scene clips produced")
        return final_path

    # Step 3: Concatenate all scene clips
    print("    Concatenating scenes...")
    video_no_audio = output_dir / "_full_video.mp4"
    _concat_videos(scene_clips, video_no_audio, width, height)

    # Step 4: Merge with audio
    print("    Merging audio...")
    _merge_audio_video(video_no_audio, audio_path, final_path)

    # Cleanup temp files
    for tmp in output_dir.glob("_*.mp4"):
        tmp.unlink(missing_ok=True)
    for tmp in output_dir.glob("_*.mp3"):
        tmp.unlink(missing_ok=True)
    for tmp in output_dir.glob("_*.txt"):
        tmp.unlink(missing_ok=True)

    print(f"    ✓ Final video: {final_path}")
    return final_path


def _compose_scene_overlay(
    avatar_path: Path,
    visual_clips: list,
    output: Path,
    duration: float,
    width: int,
    height: int,
    fps: int,
) -> None:
    """Compose a scene with avatar PIP over visual background."""
    # First build the background from visuals
    bg_clip = output.parent / f"_bg_{output.stem}.mp4"
    _compose_scene_visuals(visual_clips, bg_clip, duration, width, height, fps)

    if not bg_clip.exists():
        # Fallback to just avatar
        _scale_and_trim(avatar_path, output, duration, width, height)
        return

    # PIP: avatar in bottom-right, 25% size
    pip_w = width // 4
    pip_h = height // 4
    margin = 20

    _run_ffmpeg([
        "-i", str(bg_clip),
        "-i", str(avatar_path),
        "-filter_complex", (
            f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1[bg];"
            f"[1:v]scale={pip_w}:{pip_h}:force_original_aspect_ratio=decrease,"
            f"pad={pip_w}:{pip_h}:(ow-iw)/2:(oh-ih)/2,setsar=1[pip];"
            f"[bg][pip]overlay={width - pip_w - margin}:{height - pip_h - margin}"
            f":shortest=1[outv]"
        ),
        "-map", "[outv]",
        "-t", str(duration),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-an",
        str(output),
    ])

    bg_clip.unlink(missing_ok=True)


def _compose_scene_visuals(
    visual_clips: list,
    output: Path,
    duration: float,
    width: int,
    height: int,
    fps: int,
) -> None:
    """Sequence visual clips (animations and images) for a scene."""
    if not visual_clips:
        return

    raw_durations = [max(seg.duration_seconds, 0.1) for _, _, seg in visual_clips]
    total_raw = sum(raw_durations) or duration
    scaled_durations = [duration * (clip_duration / total_raw) for clip_duration in raw_durations]

    temp_clips = []
    for j, ((vtype, path, seg), clip_duration) in enumerate(zip(visual_clips, scaled_durations)):
        temp_path = output.parent / f"_vis_{output.stem}_{j}.mp4"

        if vtype == "animation":
            _scale_and_trim(path, temp_path, clip_duration, width, height)
        else:
            # Image — slow zoom with slight pan for visual interest
            _image_to_video_cinematic(path, temp_path, clip_duration, width, height, fps)

        if temp_path.exists():
            temp_clips.append(temp_path)

    if not temp_clips:
        return

    if len(temp_clips) == 1:
        temp_clips[0].rename(output)
        return

    _concat_videos(temp_clips, output, width, height)

    for tc in temp_clips:
        tc.unlink(missing_ok=True)


def _image_to_video_cinematic(
    image: Path, output: Path, duration: float, width: int, height: int, fps: int
) -> None:
    """Convert image to video with cinematic slow zoom + slight pan."""
    # Scale image up 20% for zoom room, then slowly zoom and pan
    total_frames = int(duration * fps)
    _run_ffmpeg([
        "-loop", "1",
        "-framerate", str(fps),
        "-i", str(image),
        "-t", str(duration),
        "-vf", (
            f"scale={int(width * 1.3)}:{int(height * 1.3)},"
            f"zoompan=z='1.0+on/{total_frames}*0.15'"
            f":x='iw/2-(iw/zoom/2)+sin(on/{total_frames}*3.14)*50'"
            f":y='ih/2-(ih/zoom/2)'"
            f":d=1:s={width}x{height},"
            f"tpad=stop_mode=clone:stop_duration=999"
        ),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-an",
        str(output),
    ])


def _scale_and_trim(
    input_path: Path, output: Path, duration: float, width: int, height: int
) -> None:
    """Scale a video to target resolution, trim to duration, and freeze last frame if source is too short."""
    _run_ffmpeg([
        "-i", str(input_path),
        "-t", str(duration),
        "-vf", (
            f"tpad=stop_mode=clone:stop_duration=999,"
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1"
        ),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-an",
        str(output),
    ])


def _create_black_clip(
    output: Path, duration: float, width: int, height: int, fps: int
) -> None:
    """Create a black video clip as placeholder."""
    _run_ffmpeg([
        "-f", "lavfi",
        "-i", f"color=c=black:s={width}x{height}:d={duration}:r={fps}",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(output),
    ])


def _create_branded_clip(
    output: Path, duration: float, width: int, height: int, fps: int
) -> None:
    """Create a dark-teal branded placeholder (never pure black)."""
    _run_ffmpeg([
        "-f", "lavfi",
        "-i", f"color=c=0x0d1b2a:s={width}x{height}:d={duration}:r={fps}",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(output),
    ])


def _concat_audio(voice_dir: Path, output_dir: Path) -> Path | None:
    """Concatenate all scene MP3s into one audio file."""
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
    _run_ffmpeg([
        "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy",
        str(output),
    ])

    return output if output.exists() else None


def _concat_videos(clips: list[Path], output: Path, width: int, height: int) -> None:
    """Concatenate video clips with uniform scaling."""
    if not clips:
        return

    inputs = []
    filter_parts = []
    for i, clip in enumerate(clips):
        inputs.extend(["-i", str(clip)])
        filter_parts.append(
            f"[{i}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1[v{i}]"
        )

    concat_inputs = "".join(f"[v{i}]" for i in range(len(clips)))
    filter_parts.append(f"{concat_inputs}concat=n={len(clips)}:v=1:a=0[outv]")

    cmd = inputs + [
        "-filter_complex", ";".join(filter_parts),
        "-map", "[outv]",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(output),
    ]
    _run_ffmpeg(cmd)


def _merge_audio_video(video: Path, audio: Path, output: Path) -> None:
    """Merge video and audio, trimming to shortest."""
    _run_ffmpeg([
        "-i", str(video),
        "-i", str(audio),
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        str(output),
    ])


def _get_duration(path: Path) -> float:
    """Get media file duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True,
        )
        return float(result.stdout.strip())
    except (ValueError, FileNotFoundError):
        return 0.0


def _extract_text_tags(script: ProductionScript) -> list[dict]:
    """Extract [TEXT:] tags from the script for overlay positioning."""
    entries = []
    cumulative_seconds = 0.0

    for scene in script.scenes:
        for match in re.finditer(r"\[TEXT:\s*([^\]]+)\]", scene.script_full):
            text = match.group(1).strip().strip('"').strip("'")
            offset_ratio = match.start() / max(len(scene.script_full), 1)
            scene_seconds = scene.duration_minutes * 60
            start = cumulative_seconds + (offset_ratio * scene_seconds)
            entries.append({
                "text": text,
                "start": start,
                "end": start + 5,
            })
        cumulative_seconds += scene.duration_minutes * 60

    return entries


def _parse_scene_number(scene_label: str) -> int:
    match = re.match(r"(\d+)", scene_label.strip())
    return int(match.group(1)) if match else 0


def _scene_media_sort_key(path: Path) -> tuple[int, str]:
    """Sort scene media numerically so scene10 comes after scene9."""
    match = re.search(r"scene(\d+)", path.name)
    return (int(match.group(1)) if match else 0, path.name)


def _run_ffmpeg(args: list[str]) -> None:
    """Run ffmpeg with given arguments, suppressing output unless error."""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[:500]}")
