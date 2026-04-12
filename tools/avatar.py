"""Generate talking-head avatar videos using Wavespeed InfiniteTalk.

Takes voice audio + reference face image → produces a lip-synced avatar video.
The API is async: submit a job, poll for completion, download the result.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import httpx

from tools.models import GeneratedAvatar, GeneratedVoice, ProductionScript
from tools.parallel import run_parallel, safe_print

WAVESPEED_BASE = "https://api.wavespeed.ai/api/v3"
POLL_INTERVAL_SECONDS = 5
MAX_POLL_ATTEMPTS = 120  # 10 minutes max

# Network reliability tuning. Wavespeed's edge intermittently drops connections
# (Errno 60 / ConnectTimeout) on large uploads + downloads. We retry with
# exponential backoff so a single TCP hiccup doesn't lose a generated clip.
NETWORK_RETRIES = int(os.environ.get("WAVESPEED_RETRIES", "5"))
NETWORK_BACKOFF = float(os.environ.get("WAVESPEED_BACKOFF", "3.0"))
UPLOAD_TIMEOUT = httpx.Timeout(600.0, connect=60.0)
DOWNLOAD_TIMEOUT = httpx.Timeout(600.0, connect=60.0)


def generate_avatars(
    script: ProductionScript,
    voices: list[GeneratedVoice],
    output_dir: Path,
    reference_image: str = "",
    resolution: str = "480p",
    skip_existing: bool = True,
) -> list[GeneratedAvatar]:
    """Generate avatar videos for all scenes that have voice audio.

    Args:
        script: The production script.
        voices: Generated voice files (need to upload these first).
        output_dir: Directory to save avatar videos.
        reference_image: URL or local path to the avatar face image.
        resolution: Video resolution ("480p", "720p").
        skip_existing: Skip scenes that already have avatar videos.

    Returns:
        List of GeneratedAvatar objects.
    """
    api_key = os.environ.get("WAVESPEED_API_KEY", "")
    if not api_key:
        raise ValueError("WAVESPEED_API_KEY not set in environment")

    if not reference_image:
        raise ValueError(
            "reference_image is required — provide a URL or local path to the avatar face image"
        )

    avatar_dir = output_dir / "avatars"
    avatar_dir.mkdir(parents=True, exist_ok=True)

    # If reference_image is a local file, upload it
    image_url = reference_image
    if not reference_image.startswith("http"):
        print("    Uploading reference image...")
        image_url = upload_media(api_key, reference_image)
        print(f"    ✓ Uploaded: {image_url[:80]}...")

    def _scene_speech_prompt(scene_num: int) -> str:
        for scene in script.scenes:
            if _parse_scene_number(scene.scene) == scene_num:
                return scene.speech_prompt or script.speech_prompt
        return script.speech_prompt

    def _one_avatar(voice: GeneratedVoice, _idx: int) -> GeneratedAvatar | None:
        scene_num = voice.scene_number
        filename = f"scene{scene_num}_avatar.mp4"
        filepath = avatar_dir / filename

        if skip_existing and filepath.exists():
            safe_print(f"    [scene {scene_num}] skip (exists): {filename}")
            return GeneratedAvatar(
                scene_number=scene_num,
                scene_title=voice.scene_title,
                file_path=str(filepath),
            )

        speech_prompt = _scene_speech_prompt(scene_num)
        safe_print(f"    [scene {scene_num}] uploading audio…")
        audio_url = upload_media(api_key, voice.file_path)

        safe_print(f"    [scene {scene_num}] submitting to InfiniteTalk…")
        video_url = _create_and_poll(
            api_key=api_key,
            audio_url=audio_url,
            image_url=image_url,
            prompt=speech_prompt,
            resolution=resolution,
            scene_label=f"scene {scene_num}",
        )

        safe_print(f"    [scene {scene_num}] downloading video…")
        _download_file(video_url, filepath)

        safe_print(f"    [scene {scene_num}] ✓ saved {filename}")
        return GeneratedAvatar(
            scene_number=scene_num,
            scene_title=voice.scene_title,
            file_path=str(filepath),
            audio_url=audio_url,
            video_url=video_url,
        )

    results = run_parallel(
        voices,
        _one_avatar,
        max_workers=int(os.environ.get("AVATAR_PARALLEL", "3")),
        label="avatars",
    )
    return [r.value for r in results if r.ok and r.value is not None]


def upload_media(api_key: str, file_path: str) -> str:
    """Upload a local file to Wavespeed and return the hosted URL.

    Retries on transient network failures with exponential backoff so a single
    Errno 60 / ConnectTimeout doesn't lose the asset.
    """
    last_err: Exception | None = None
    response = None
    for attempt in range(1, NETWORK_RETRIES + 1):
        try:
            with open(file_path, "rb") as f:
                response = httpx.post(
                    f"{WAVESPEED_BASE}/media/upload/binary",
                    headers={"Authorization": f"Bearer {api_key}"},
                    files={"file": f},
                    timeout=UPLOAD_TIMEOUT,
                )
            break
        except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError,
                httpx.RemoteProtocolError, OSError) as e:
            last_err = e
            if attempt == NETWORK_RETRIES:
                raise RuntimeError(
                    f"Upload failed after {NETWORK_RETRIES} retries: {e}"
                ) from e
            sleep_for = NETWORK_BACKOFF * (2 ** (attempt - 1))
            print(f"      ↻ Upload retry {attempt}/{NETWORK_RETRIES} in {sleep_for:.0f}s ({e})")
            time.sleep(sleep_for)

    if response is None or response.status_code != 200:
        status = response.status_code if response is not None else "no-response"
        body = response.text[:300] if response is not None else str(last_err)
        raise RuntimeError(f"Upload failed ({status}): {body}")

    data = response.json()
    # Extract the download URL from the nested response
    if isinstance(data, dict):
        # Try common response shapes
        if "data" in data:
            inner = data["data"]
            if isinstance(inner, dict):
                # Look for URL fields
                for key in ("download_url", "url", "file_url", "media_url"):
                    if key in inner and isinstance(inner[key], str):
                        return inner[key]
            elif isinstance(inner, str) and inner.startswith("http"):
                return inner
        # Top-level URL fields
        for key in ("download_url", "url", "file_url"):
            if key in data and isinstance(data[key], str):
                return data[key]

    raise RuntimeError(f"Could not extract URL from upload response: {data}")


def _create_and_poll(
    api_key: str,
    audio_url: str,
    image_url: str,
    prompt: str,
    resolution: str,
    scene_label: str = "",
) -> str:
    """Submit an InfiniteTalk job and poll until complete. Returns video URL."""
    headers = {"Authorization": f"Bearer {api_key}"}

    # Submit job — use infinitetalk-fast for better lip sync
    response = httpx.post(
        f"{WAVESPEED_BASE}/wavespeed-ai/infinitetalk",
        headers=headers,
        json={
            "audio": audio_url,
            "image": image_url,
            "prompt": prompt,
            "resolution": resolution,
        },
        timeout=60.0,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"InfiniteTalk submit failed ({response.status_code}): {response.text[:300]}"
        )

    resp_data = response.json()
    poll_url = resp_data["data"]["urls"]["get"]

    # Poll for completion with retry on connection failures
    for attempt in range(MAX_POLL_ATTEMPTS):
        time.sleep(POLL_INTERVAL_SECONDS)

        try:
            poll_resp = httpx.get(
                poll_url, headers=headers,
                timeout=httpx.Timeout(60.0, connect=30.0),
            )
        except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError) as e:
            if attempt % 6 == 5:
                safe_print(f"      {scene_label} connection retry… "
                           f"({(attempt + 1) * POLL_INTERVAL_SECONDS}s)")
            continue

        if poll_resp.status_code != 200:
            continue

        data = poll_resp.json().get("data", {})
        status = data.get("status", "")

        if status == "completed":
            outputs = data.get("outputs", [])
            if outputs:
                return outputs[0]
            raise RuntimeError("InfiniteTalk completed but no outputs returned")

        if status == "failed" or status == "error":
            raise RuntimeError(f"InfiniteTalk job failed: {data}")

        # Still processing
        if attempt % 6 == 5:
            safe_print(f"      {scene_label} still processing… "
                       f"({(attempt + 1) * POLL_INTERVAL_SECONDS}s)")

    raise TimeoutError(
        f"InfiniteTalk job timed out after {MAX_POLL_ATTEMPTS * POLL_INTERVAL_SECONDS}s"
    )


def _download_file(url: str, filepath: Path) -> None:
    """Download a file from URL to local path with retry on transient failures.

    Wavespeed generates the clip server-side fine but the CDN occasionally
    closes the connection mid-download — we retry instead of losing the asset.
    """
    tmp_path = filepath.with_suffix(filepath.suffix + ".part")
    last_err: Exception | None = None
    for attempt in range(1, NETWORK_RETRIES + 1):
        try:
            with httpx.stream(
                "GET", url, timeout=DOWNLOAD_TIMEOUT, follow_redirects=True,
            ) as response:
                response.raise_for_status()
                with open(tmp_path, "wb") as f:
                    for chunk in response.iter_bytes(chunk_size=65536):
                        f.write(chunk)
            tmp_path.replace(filepath)
            return
        except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError,
                httpx.RemoteProtocolError, httpx.HTTPStatusError, OSError) as e:
            last_err = e
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            if attempt == NETWORK_RETRIES:
                raise RuntimeError(
                    f"Download failed after {NETWORK_RETRIES} retries: {e}"
                ) from e
            sleep_for = NETWORK_BACKOFF * (2 ** (attempt - 1))
            print(f"      ↻ Download retry {attempt}/{NETWORK_RETRIES} in {sleep_for:.0f}s ({e})")
            time.sleep(sleep_for)


def _parse_scene_number(scene_label: str) -> int:
    """Extract scene number from labels like '3 - The Mechanism'."""
    match = re.match(r"(\d+)", scene_label.strip())
    return int(match.group(1)) if match else 0
