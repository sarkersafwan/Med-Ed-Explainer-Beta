"""CLI entry point for the medical education video pipeline.

Usage:
    python run.py                                    # Interactive: choose PDF or topic
    python run.py input.pdf                          # Use a specific PDF
    python run.py --topic "Muscle Contraction"       # Generate from topic (no PDF)
    python run.py input.pdf --dry-run                # Generate script locally only
    python run.py input.pdf --duration 8             # Skip duration prompt, use 8 min
    python run.py input.pdf --images-only            # Generate images from existing script
    python run.py input.pdf --voice-only             # Generate voice from existing script
    python run.py input.pdf --skip-images            # Skip image generation
    python run.py input.pdf --skip-voice             # Skip voice generation
    python run.py input.pdf --skip-avatar            # Skip avatar video generation
    python run.py input.pdf --skip-animation         # Skip animation video generation
    python run.py input.pdf --voice-id XXX           # Override ElevenLabs voice
    python run.py input.pdf --avatar-image face.jpg  # Reference face for avatar
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from tools.analyze import analyze_content, rebuild_scenes_for_duration
from tools.animations import generate_animations_from_segments
from tools.avatar import generate_avatars
from tools.character_sheet import (
    build_character_spec,
    character_is_needed,
    generate_character_sheet,
)
from tools.compose import compose_video
from tools.compose_remotion import compose_with_remotion
from tools.creative_brief import build_brief_from_inputs
from tools.extract import extract_pdf
from tools.generate_content import generate_content_from_topic
from tools.generate_images import generate_images_from_segments
from tools.generate_script import generate_script
from tools.generate_segments import generate_segments
from tools.generate_voice import generate_voice, list_voices
from tools.models import CharacterSpec, ProductionScript, RunManifest, Segment
from tools.project_store import (
    DEFAULT_MAX_PROJECT_RUNS,
    RunContext,
    create_run_context,
    export_latest_character_assets,
    get_run_dir,
    prune_project_runs,
    resolve_existing_script_path,
    set_latest_run,
    slugify_project_name,
    write_project_manifest,
)
from tools.quality import validate_script
from tools.review import (
    build_content_evidence,
    build_scene_evidence,
    review_script_against_evidence,
    write_review_artifacts,
)


def _time_box_script(script: "ProductionScript", target_minutes: float) -> None:
    """Trim a script in-place so it fits a user-specified duration budget.

    Script generators tend to overshoot on short videos — you ask for 30s
    and get 46s. This function enforces the budget in two stages:

      1. Per-scene word trim at 150 WPM ceiling. If a scene has more words
         than its duration allows, keep the first N words at a sentence
         boundary and update both `script` and `script_full`.
      2. If the total still overshoots by more than 20%, drop trailing
         scenes until we're inside budget (always keep at least 2 scenes).

    Logs what it trimmed so the user sees what content got cut.
    """
    import re as _re
    WPM_CEILING = 170  # matches quality.py upper band
    total_budget_words = max(10, int(round(target_minutes * WPM_CEILING)))

    def _strip_tags_local(text: str) -> str:
        return _re.sub(r"\[[^\]]+\]", "", text)

    # Stage 1: per-scene trim
    trimmed_any = False
    for scene in script.scenes:
        scene_budget = max(
            5, int(round(scene.duration_minutes * WPM_CEILING))
        )
        clean = _strip_tags_local(scene.script_full)
        words = clean.split()
        if len(words) <= scene_budget:
            continue

        # Trim to budget, then back off to a sentence boundary so the cut
        # doesn't land mid-thought.
        cut = " ".join(words[:scene_budget])
        # Find last sentence-ending punctuation.
        m = _re.search(r"[.!?](?!.*[.!?])", cut)
        if m:
            cut = cut[: m.end()]

        original_words = len(words)
        new_words = len(cut.split())
        print(f"   ✂️  Scene '{scene.scene}': trimmed "
              f"{original_words} → {new_words} words to fit {scene.duration_minutes} min")
        # Preserve leading tags from script_full so MODE/VISUAL etc. remain.
        leading_tags = ""
        tag_match = _re.match(r"((?:\[[^\]]+\]\s*)+)", scene.script_full)
        if tag_match:
            leading_tags = tag_match.group(1)
        scene.script_full = (leading_tags + cut).strip()
        scene.script = cut
        scene.word_count = new_words
        trimmed_any = True

    # Stage 2: drop trailing scenes if total still overshoots by >20%
    total_words = sum(s.word_count for s in script.scenes)
    overshoot_limit = int(total_budget_words * 1.2)
    while total_words > overshoot_limit and len(script.scenes) > 2:
        dropped = script.scenes.pop()
        total_words -= dropped.word_count
        print(f"   ✂️  Dropped trailing scene '{dropped.scene}' "
              f"({dropped.word_count} words) — still over budget")
        trimmed_any = True

    if trimmed_any:
        script.total_word_count = sum(s.word_count for s in script.scenes)
        script.total_minutes = round(
            sum(s.duration_minutes for s in script.scenes), 2
        )
        print(f"   ✓ Time-boxed: {script.total_word_count} words across "
              f"{len(script.scenes)} scenes for {target_minutes} min budget")


def _keep_system_awake() -> object | None:
    """Prevent the Mac from sleeping during a long pipeline run.

    macOS sleep stalls network polling (Wavespeed, Kling, Gemini) which
    silently breaks in-flight generations. We spawn `caffeinate` as a child
    process scoped to our PID — it exits automatically when we exit. Returns
    the Popen object so the caller holds a reference (GC wouldn't kill it
    anyway, but keeps intent explicit). Returns None on non-macOS or if
    caffeinate isn't available.
    """
    import sys
    import subprocess
    if sys.platform != "darwin":
        return None
    try:
        proc = subprocess.Popen(
            ["caffeinate", "-i", "-m", "-s", "-w", str(os.getpid())],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("☕ caffeinate running — system will stay awake for this run")
        return proc
    except (FileNotFoundError, OSError):
        return None


def main() -> None:
    # Start caffeinate first so any slow setup steps are also protected.
    _caffeinate_proc = _keep_system_awake()

    parser = argparse.ArgumentParser(
        description="Transform medical education content into AI explainer video scripts"
    )
    parser.add_argument("pdf", nargs="?", default=None, help="Path to the input PDF file (optional — will prompt if not provided)")
    parser.add_argument(
        "--topic",
        type=str,
        default="",
        help="Generate from a topic string instead of a PDF",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate script locally without pushing to Airtable",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Target video duration in minutes (skips interactive prompt)",
    )
    parser.add_argument(
        "--images-only",
        action="store_true",
        help="Generate images from an existing script (skip script generation)",
    )
    parser.add_argument(
        "--voice-only",
        action="store_true",
        help="Generate voice from an existing script (skip script generation)",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Skip image generation",
    )
    parser.add_argument(
        "--skip-voice",
        action="store_true",
        help="Skip voice generation",
    )
    parser.add_argument(
        "--voice-id",
        type=str,
        default="",
        help="ElevenLabs voice ID (overrides ELEVENLABS_VOICE_ID env var)",
    )
    parser.add_argument(
        "--skip-avatar",
        action="store_true",
        help="Skip avatar video generation",
    )
    parser.add_argument(
        "--skip-animation",
        action="store_true",
        help="Skip animation video generation",
    )
    parser.add_argument(
        "--avatar-image",
        type=str,
        default="",
        help="URL or local path to the avatar reference face image",
    )
    parser.add_argument(
        "--character-image",
        type=str,
        default="",
        help="Local path to a headshot to use as the patient character reference "
             "(overrides auto-generated character sheet)",
    )
    parser.add_argument(
        "--skip-compose",
        action="store_true",
        help="Skip final video composition",
    )
    parser.add_argument(
        "--brief",
        type=str,
        default="",
        help="Full creative brief text (replaces --topic with rich direction)",
    )
    parser.add_argument(
        "--style-image",
        type=str,
        default="",
        help="Style reference image — guides the visual aesthetic of generated images",
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List available ElevenLabs voices and exit",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "creative", "production"],
        default=os.environ.get("PIPELINE_MODE", "auto"),
        help="Pipeline safety mode. Auto => production for PDF input, creative for topic input.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Existing run ID to target for --images-only / --voice-only recovery flows.",
    )
    args = parser.parse_args()

    # Handle --list-voices early exit
    if args.list_voices:
        _list_voices_and_exit()
        return

    # --- Pre-flight validation ---
    _preflight_check(args)

    # --- Build creative brief ---
    brief = None
    if args.brief or args.style_image:
        print(f"\n🎨 Building creative brief...")
        brief = build_brief_from_inputs(
            topic=args.topic,
            request_text=args.brief,
            duration=args.duration or 0,
            style_image_path=args.style_image,
        )
        if brief.topic and not args.topic:
            args.topic = brief.topic
        if brief.duration_minutes and not args.duration:
            args.duration = brief.duration_minutes
        print(f"   Topic: {brief.topic}")
        if brief.voice_direction:
            print(f"   Voice: {brief.voice_direction}")
        if brief.visual_style:
            print(f"   Style: {brief.visual_style}")
        if brief.style_analysis:
            print(f"   Reference: {brief.style_analysis[:80]}...")

    # --- Determine input source: PDF or topic ---
    content = None
    source_label = ""
    source_kind = ""

    if args.topic:
        # Topic provided via flag or extracted from brief
        print(f"\n🧠 Generating medical content for: {args.topic}")
        content = generate_content_from_topic(args.topic)
        source_label = f"topic: {args.topic}"
        source_kind = "topic"
    elif args.pdf:
        # PDF provided as argument
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            print(f"Error: File not found: {pdf_path}")
            sys.exit(1)

        # Check for --images-only or --voice-only (load existing script)
        if args.images_only or args.voice_only:
            script = _load_existing_script(pdf_path, run_id=args.run_id)
            if not script:
                sys.exit(1)
            output_dir = _resolve_script_output_dir(script)
            output_dir.mkdir(parents=True, exist_ok=True)
            if args.images_only:
                # Load or regenerate segments
                segments_path = output_dir / "segments.json"
                if segments_path.exists():
                    seg_data = json.loads(segments_path.read_text())
                    segs = [Segment(**s) for s in seg_data]
                else:
                    segs = generate_segments(script)
                # Load existing character sheet if present
                char = None
                char_json = output_dir / "character" / "character.json"
                if char_json.exists():
                    char = CharacterSpec(**json.loads(char_json.read_text()))
                _run_image_generation_segments(segs, output_dir, character=char)
            if args.voice_only:
                _run_voice_generation(script, output_dir, args.voice_id)
            print(f"\n✨ Done!")
            return

        print(f"\n📄 Extracting content from {pdf_path.name}...")
        content = extract_pdf(pdf_path)
        source_label = f"pdf: {pdf_path.name}"
        source_kind = "pdf"
    else:
        # No input — ask interactively
        content, source_label = _interactive_source_picker()
        source_kind = "pdf" if source_label.startswith("pdf:") else "topic"

    print(f"   Topic: {content.topic}")
    print(f"   Subject: {content.subject} | System: {content.system}")
    if content.clinical_vignette:
        print(f"   Vignette: {content.clinical_vignette[:80]}...")
    print(f"   Answer choices: {len(content.answer_choices)}")
    if content.correct_answer:
        print(f"   Correct: {content.correct_answer_letter}. {content.correct_answer}")

    pipeline_mode = _resolve_pipeline_mode(args.mode, source_kind)
    grounded = source_kind == "pdf"
    print(f"   Mode: {pipeline_mode} ({'grounded' if grounded else 'ungrounded'})")

    # Step 2: Analyze
    print(f"\n🔬 Analyzing content...")
    plan = analyze_content(content)
    print(f"   Complexity: {plan.complexity_score}/5")
    print(f"   Core concepts: {plan.concept_count}")
    print(f"   Differentials: {plan.differential_count}")

    # Step 3: Duration selection
    if args.duration:
        target_minutes = args.duration
        print(f"\n⏱️  Using specified duration: {target_minutes} min")
    else:
        target_minutes = _interactive_duration_picker(plan)

    # Rebuild scenes if duration differs from recommended
    scenes = plan.scenes
    if abs(target_minutes - plan.recommended_minutes) > 0.5:
        scenes = rebuild_scenes_for_duration(content, plan, target_minutes)
        print(f"   Rebuilt scene plan: {len(scenes)} scenes for {target_minutes} min")

    print(f"\n📝 Scene plan:")
    for s in scenes:
        print(f"   {s.scene_number}. [{s.purpose}] {s.scene_title} ({s.estimated_minutes} min)")

    # Step 4: Generate script
    print(f"\n🎬 Generating production script...")
    creative_direction = brief.to_script_prompt() if brief else ""
    script = generate_script(content, plan, target_minutes=target_minutes, scenes=scenes, creative_direction=creative_direction)
    print(f"   Generated: {len(script.scenes)} scenes, {script.total_word_count} words, {script.total_minutes} min")

    # Enforce the time-box when the user explicitly set --duration.
    # The generator tends to overshoot on short videos; trim each scene to
    # its word budget at 150 WPM and drop trailing scenes if total still
    # overshoots by more than 20%. Without this, a 30s-requested video can
    # come back as 46s.
    if args.duration:
        _time_box_script(script, target_minutes)

    # Step 4b: Create run-scoped output context and persist metadata
    run_context = create_run_context(script.project_name)
    output_dir = run_context.run_dir
    script.run_id = run_context.run_id
    script.pipeline_mode = pipeline_mode
    script.source_kind = source_kind
    script.grounded = grounded
    script.created_at = datetime.now().astimezone()

    # Step 5: Quality check
    print(f"\n✅ Running quality checks...")
    issues = validate_script(script)

    # When the user explicitly passed --duration, they're time-boxing the
    # video and we accept that the script may run "hot" (high WPM) or
    # truncate content to fit. Demote WPM / total-duration / scene-length
    # issues to warnings in that case instead of letting them halt the
    # pipeline in production mode. Real blockers (missing MODE tags, invalid
    # mode values, AI slop phrases, etc.) still block as before.
    duration_user_specified = bool(args.duration)
    demotable_markers = (
        "Word count too low for duration",
        "Word count too high for duration",
        "Duration exceeds",
        "Script too short",
        "Scene durations (",
    )
    demoted: list[str] = []
    remaining: list[str] = []
    if duration_user_specified and issues:
        for issue in issues:
            if any(marker in issue for marker in demotable_markers):
                demoted.append(issue)
            else:
                remaining.append(issue)
        issues = remaining

    if issues:
        print(f"   ⚠️  Found {len(issues)} issue(s):")
        for issue in issues:
            print(f"      - {issue}")
    else:
        print(f"   All checks passed!")
    if demoted:
        print(f"   ℹ️  {len(demoted)} timing issue(s) accepted (you specified --duration):")
        for issue in demoted:
            print(f"      - {issue}")

    # Step 5b: Grounding/evidence review
    print(f"\n🧾 Building evidence + review artifacts...")
    evidence_sections = build_content_evidence(content, grounded=grounded)
    scene_evidence = build_scene_evidence(content, scenes, grounded=grounded)
    script_review = review_script_against_evidence(
        script,
        scene_evidence,
        grounded=grounded,
        pipeline_mode=pipeline_mode,
    )
    write_review_artifacts(
        evidence_sections,
        scene_evidence,
        script_review,
        output_dir=output_dir,
    )
    print(f"   Review: {'approved' if script_review.approved else 'blocked'}")
    if script_review.summary:
        print(f"   Summary: {script_review.summary}")
    for blocker in script_review.blockers:
        print(f"   Blocker: {blocker}")
    for warning in script_review.warnings:
        print(f"   Warning: {warning}")

    # Step 6: Save locally
    script_path = output_dir / "script.json"
    plan_path = output_dir / "plan.json"
    manifest_path = output_dir / "run_manifest.json"
    script.source_pdf = source_label
    script_path.write_text(json.dumps(script.model_dump(mode="json"), indent=2))
    plan_path.write_text(json.dumps({
        "topic": plan.topic,
        "complexity_score": plan.complexity_score,
        "recommended_minutes": plan.recommended_minutes,
        "duration_options": [opt.model_dump() for opt in plan.duration_options],
        "scenes": [scene.model_dump() for scene in scenes],
    }, indent=2))
    manifest = RunManifest(
        project_name=script.project_name,
        run_id=run_context.run_id,
        pipeline_mode=pipeline_mode,
        source_kind=source_kind,
        grounded=grounded,
        source_label=source_label,
        target_minutes=target_minutes,
        created_at=script.created_at,
        script_review_approved=script_review.approved and not issues,
        script_review_summary=script_review.summary,
    )
    manifest_path.write_text(json.dumps(manifest.model_dump(), indent=2, default=str))
    write_project_manifest(run_context.project_dir, script.project_name, run_context.run_id)
    set_latest_run(run_context.project_dir, run_context.run_id)
    pruned_runs = prune_project_runs(run_context.project_dir, keep=DEFAULT_MAX_PROJECT_RUNS)
    print(f"\n💾 Saved to {script_path}")
    print(f"   Run folder: {output_dir}")
    print(f"   Project folder: {run_context.project_dir}")
    if pruned_runs:
        print(f"   Pruned older runs: {', '.join(run_dir.name for run_dir in pruned_runs)}")

    blocking_issues = list(script_review.blockers)
    if pipeline_mode == "production" and issues:
        blocking_issues.extend(issues)

    if pipeline_mode == "production" and (blocking_issues or not script_review.approved):
        print(f"\n🛑 Production mode stopped before asset generation.")
        print(f"   Review the saved artifacts in {output_dir}")
        print(f"   Fix the blockers or rerun in creative mode for exploratory work.")
        return

    # Print script preview
    print(f"\n{'='*60}")
    print(f"SCRIPT PREVIEW: {script.topic}")
    print(f"{'='*60}")
    for scene in script.scenes:
        print(f"\n--- {scene.scene} ({scene.duration_minutes} min, {scene.word_count} words) ---")
        preview = scene.script_full[:300]
        if len(scene.script_full) > 300:
            preview += "..."
        print(preview)
    print(f"\n{'='*60}")

    # --- Segment Generation ---

    character_seed: CharacterSpec | None = None
    if not args.skip_images and content.clinical_vignette:
        try:
            character_seed = build_character_spec(content, script, [])
            if character_seed.one_line:
                print(f"\n🧬 Character lock seed: {character_seed.one_line}")
        except Exception as e:
            print(f"\n🧬 Character lock seed unavailable ({e}) — continuing without it")

    print(f"\n🧩 Generating visual segments...")
    style_direction = brief.to_image_style_prompt() if brief else ""
    segments = generate_segments(script, style_direction=style_direction, character=character_seed)
    print(f"   Generated {len(segments)} segments across {len(script.scenes)} scenes")

    # Save segments
    segments_path = output_dir / "segments.json"
    segments_path.write_text(json.dumps([s.model_dump() for s in segments], indent=2))

    # --- Filter out avatar-mode segments FIRST ---
    # Avatar-mode scenes show the talking head full-screen and don't need
    # image/animation assets. Compute this before the character-sheet gate so
    # we only build a character if a surviving segment actually needs it.
    import re as _re
    def _scene_num_from_label(label: str) -> int:
        m = _re.match(r"(\d+)", label.strip())
        return int(m.group(1)) if m else 0
    avatar_scene_nums = {
        _scene_num_from_label(s.scene)
        for s in script.scenes
        if "[MODE: avatar]" in s.script_full
    }
    visual_segments = [s for s in segments if s.scene_number not in avatar_scene_nums]
    if avatar_scene_nums:
        skipped = len(segments) - len(visual_segments)
        if skipped:
            print(f"\n🎯 Skipping {skipped} segment(s) for avatar-mode scenes "
                  f"(scenes {sorted(avatar_scene_nums)}) — saves Gemini + Kling spend")

    # --- Character Sheet (smart gate — only if a visual segment will use it) ---

    character: CharacterSpec | None = character_seed
    if not args.skip_images:
        needs_character = bool(args.character_image) or character_is_needed(visual_segments)
        if needs_character:
            print(f"\n🧑 Building character sheet...")
            try:
                if character is None:
                    character = build_character_spec(content, script, visual_segments)
                print(f"   Spec: {character.one_line}")
                generate_character_sheet(character, output_dir, override_image=args.character_image)
                exported = export_latest_character_assets(run_context)
                if exported:
                    print(f"   Project character folder: {exported}")
            except Exception as e:
                print(f"   ⚠️  Character sheet failed ({e}) — continuing without reference")
                character = None
        else:
            print(f"\n🧑 No visual segments with human intent — skipping character sheet "
                  f"(would have been wasted)")

    # Phase 2 stages (image gen + voice gen) have no mutual dependency —
    # run them concurrently so the user doesn't wait for voices before images
    # even start. Same for Phase 3 (avatar + animation).
    from tools.parallel import run_stages_in_parallel

    phase2_stages: list[tuple[str, callable]] = []
    if not args.skip_images:
        phase2_stages.append(
            ("images", lambda: _run_image_generation_segments(
                visual_segments, output_dir, character=character))
        )
    if not args.skip_voice:
        phase2_stages.append(
            ("voices", lambda: _run_voice_generation(script, output_dir, args.voice_id))
        )
    if phase2_stages:
        run_stages_in_parallel(phase2_stages)

    # --- Phase 3: Avatar + Animation ---
    phase3_stages: list[tuple[str, callable]] = []
    if not args.skip_avatar and not args.skip_voice:
        phase3_stages.append(
            ("avatars", lambda: _run_avatar_generation(
                script, output_dir, args.voice_id, args.avatar_image))
        )
    if not args.skip_animation and not args.skip_images:
        phase3_stages.append(
            ("animations", lambda: _run_animation_generation_segments(
                visual_segments, output_dir))
        )
    if phase3_stages:
        run_stages_in_parallel(phase3_stages)

    # --- Checkpoint: Review before composition ---

    if not args.skip_compose and not args.dry_run:
        print(f"\n📋 Assets ready for review before final composition.")
        print(f"   Review the run folder now — verify images, voice, avatar, and animations look good.")
        try:
            proceed = input("   Ready to compose final video? [y/n] (default: y): ").strip().lower()
        except EOFError:
            proceed = "y"
        if proceed == "n":
            print(f"   Skipping composition. Run again with existing assets to compose later.")
        else:
            print(f"\n🎬 Phase 4: Video Composition (Remotion + ffmpeg)")
            try:
                final_path = compose_with_remotion(script, segments, output_dir)
                if final_path.exists():
                    print(f"   ✓ Final video: {final_path}")
                    print(f"   Size: {final_path.stat().st_size / (1024*1024):.1f} MB")
                else:
                    print(f"   ⚠️  Composition produced no output")
            except Exception as e:
                print(f"   ⚠️  Remotion failed ({e}), falling back to ffmpeg...")
                try:
                    final_path = compose_video(script, segments, output_dir)
                    if final_path.exists():
                        print(f"   ✓ Final video (ffmpeg): {final_path}")
                except Exception as e2:
                    print(f"   ⚠️  Composition failed: {e2}")
    elif args.skip_compose:
        print(f"\n🏃 Skipping video composition")

    # Step: Push to Airtable (unless dry run)
    if not args.dry_run:
        print(f"\n📤 Pushing to Airtable...")
        try:
            from tools.airtable_client import AirtableClient
            from tools.avatar import upload_media

            client = AirtableClient()

            # Ensure the Projects table has the character_image attachment field
            # (idempotent, safe to call every run; warns if PAT lacks schema scope).
            character_png = output_dir / "character" / "character.png"
            if character_png.exists():
                client.ensure_project_field("character_image", "multipleAttachments")

            # Upload avatar + style + character images for Airtable if available
            avatar_url = ""
            style_url = ""
            character_url = ""
            wavespeed_key = os.environ.get("WAVESPEED_API_KEY", "")
            if wavespeed_key:
                if args.avatar_image and Path(args.avatar_image).exists():
                    try:
                        avatar_url = upload_media(wavespeed_key, args.avatar_image)
                    except Exception:
                        pass
                if args.style_image and Path(args.style_image).exists():
                    try:
                        style_url = upload_media(wavespeed_key, args.style_image)
                    except Exception:
                        pass
                if character_png.exists():
                    try:
                        character_url = upload_media(wavespeed_key, str(character_png))
                    except Exception as e:
                        print(f"      ⚠️  Character sheet upload failed: {e}")

            project_id = client.create_project(
                script,
                voice_id=args.voice_id or os.environ.get("ELEVENLABS_VOICE_ID", ""),
                input_request=brief.to_airtable_request() if brief else f"Topic: {script.topic}",
                avatar_image_url=avatar_url,
                style_image_url=style_url,
                character_image_url=character_url,
            )
            scene_ids = client.push_scenes(script)
            print(f"   Project: {project_id}")
            print(f"   Scenes: {len(scene_ids)}")

            # Push segments — only visual segments (avatar-mode scenes don't
            # produce image/animation assets so empty segment rows are noise).
            segment_ids = []
            if visual_segments:
                segment_ids = client.push_segments(script.project_name, visual_segments)
                print(f"   Segments: {len(segment_ids)}")

            # Upload assets as attachments
            wavespeed_key = os.environ.get("WAVESPEED_API_KEY", "")
            if wavespeed_key:
                print(f"   Uploading assets to Airtable...")
                _upload_assets_to_airtable(
                    client, wavespeed_key, script, scene_ids,
                    visual_segments, segment_ids,
                    output_dir,
                )
        except Exception as e:
            print(f"   ⚠️  Airtable push failed: {e}")
            print(f"   Script saved locally at {script_path}")
    else:
        print(f"\n🏃 Dry run — skipping Airtable push")

    print(f"\n✨ Done! Assets saved to {output_dir}")


def _load_existing_script(pdf_path: Path, run_id: str = "") -> ProductionScript | None:
    """Load an existing script.json from the output directory.

    Tries to find the script by deriving the project name from the PDF,
    or by scanning the output directory.
    """
    # Try to find by extracting content and matching project name
    from tools.extract import extract_pdf

    try:
        content = extract_pdf(pdf_path)
        project_name = slugify_project_name(content.topic)
        script_path = resolve_existing_script_path(project_name, run_id=run_id)
        if script_path and script_path.exists():
            print(f"\n📂 Loading existing script from {script_path}")
            data = json.loads(script_path.read_text())
            return ProductionScript(**data)
    except Exception:
        pass

    # Fallback: scan output dir for any script.json
    output_dir = Path("output")
    if output_dir.exists():
        for script_file in sorted(output_dir.rglob("script.json"), reverse=True):
            print(f"\n📂 Loading existing script from {script_file}")
            data = json.loads(script_file.read_text())
            return ProductionScript(**data)

    print("Error: No existing script found. Run the full pipeline first.")
    return None


def _resolve_pipeline_mode(requested_mode: str, source_kind: str) -> str:
    """Resolve the effective pipeline mode."""
    if requested_mode != "auto":
        return requested_mode
    return "production" if source_kind == "pdf" else "creative"


def _resolve_script_output_dir(script: ProductionScript) -> Path:
    """Resolve the output directory for an existing script, including run-scoped paths."""
    if script.run_id:
        candidate = get_run_dir(script.project_name, script.run_id)
        if candidate.exists():
            return candidate

    candidate = resolve_existing_script_path(script.project_name, run_id=script.run_id)
    if candidate:
        return candidate.parent

    return Path("output") / script.project_name


def _scene_media_sort_key(path: Path) -> tuple[int, str]:
    """Sort scene media numerically so scene10 follows scene9."""
    import re

    match = re.search(r"scene(\d+)", path.name)
    return (int(match.group(1)) if match else 0, path.name)


def _run_image_generation_segments(
    segments: list[Segment],
    output_dir: Path,
    character: CharacterSpec | None = None,
) -> None:
    """Phase 2a: Generate hyperreal images from segments."""
    print(f"\n🎨 Phase 2: Image Generation (Hyperreal)")
    if not segments:
        print(f"   ⚠️  No segments — skipping image generation")
        return

    for seg in segments:
        print(f"   Scene {seg.scene_number}, seg {seg.segment_index}: {seg.segment_title}")

    images = generate_images_from_segments(segments, output_dir, character=character)
    print(f"\n   Generated {len(images)} images")


def _run_voice_generation(
    script: ProductionScript, output_dir: Path, voice_id: str = ""
) -> list:
    """Phase 2b: Generate TTS voice audio for each scene."""
    print(f"\n🎙️  Phase 2: Voice Generation")
    voices = generate_voice(script, output_dir, voice_id=voice_id)
    print(f"\n   Generated {len(voices)} voice files")
    return voices


def _run_avatar_generation(
    script: ProductionScript,
    output_dir: Path,
    voice_id: str = "",
    avatar_image: str = "",
) -> None:
    """Phase 3a: Generate talking-head avatar videos."""
    print(f"\n🎥 Phase 3: Avatar Generation (Wavespeed InfiniteTalk)")

    if not avatar_image:
        print("   ⚠️  No --avatar-image provided, skipping avatar generation")
        print("   Provide a face reference image: --avatar-image path/to/face.jpg")
        return

    # Load existing voice files
    from tools.models import GeneratedVoice
    voice_dir = output_dir / "voice"
    if not voice_dir.exists():
        print("   ⚠️  No voice files found — run voice generation first")
        return

    voices = []
    for mp3 in sorted(voice_dir.glob("scene*.mp3"), key=_scene_media_sort_key):
        import re
        match = re.match(r"scene(\d+)\.mp3", mp3.name)
        if match:
            scene_num = int(match.group(1))
            scene_title = ""
            for s in script.scenes:
                if s.scene.startswith(str(scene_num)):
                    scene_title = s.scene
                    break
            voices.append(GeneratedVoice(
                scene_number=scene_num,
                scene_title=scene_title,
                file_path=str(mp3),
                voice_id=voice_id,
            ))

    if not voices:
        print("   ⚠️  No voice files found in output")
        return

    avatars = generate_avatars(script, voices, output_dir, reference_image=avatar_image)
    print(f"\n   Generated {len(avatars)} avatar videos")


def _run_animation_generation_segments(segments: list[Segment], output_dir: Path) -> None:
    """Phase 3b: Generate medical animation videos from segment images."""
    print(f"\n🎬 Phase 3: Animation Generation (KIE.ai Kling 3.0)")

    images_dir = output_dir / "images"
    if not images_dir.exists():
        print("   ⚠️  No images found — run image generation first")
        return

    # Filter to segments that have images
    available = [s for s in segments if (images_dir / f"scene{s.scene_number}_seg{s.segment_index}.png").exists()]
    if not available:
        print("   ⚠️  No segment images found to animate")
        return

    print(f"   Found {len(available)} segment images to animate")
    animations = generate_animations_from_segments(available, images_dir, output_dir)
    print(f"\n   Generated {len(animations)} animation videos")


def _upload_assets_to_airtable(
    client,
    wavespeed_key: str,
    script: ProductionScript,
    scene_ids: list[str],
    segments: list[Segment],
    segment_ids: list[str],
    output_dir: Path,
) -> None:
    """Upload generated assets as attachments to Airtable records in parallel.

    Each upload is an independent (file → Wavespeed → Airtable) chain, so we
    fan them out across a thread pool. One slow/retry-heavy upload no longer
    stalls the rest of the batch.
    """
    from tools.avatar import upload_media
    from tools.parallel import run_parallel, safe_print
    import re

    def _scene_num(label: str) -> int:
        m = re.match(r"(\d+)", label.strip())
        return int(m.group(1)) if m else 0

    # Build a flat list of upload jobs first so everything runs through one
    # thread pool with a shared concurrency cap.
    jobs: list[dict] = []

    for i, scene in enumerate(script.scenes):
        if i >= len(scene_ids):
            break
        record_id = scene_ids[i]
        scene_num = _scene_num(scene.scene)

        voice_path = output_dir / "voice" / f"scene{scene_num}.mp3"
        if voice_path.exists():
            jobs.append({
                "kind": "scene_voice", "path": voice_path, "record_id": record_id,
                "scene_num": scene_num, "status_field": "Status_voice",
            })
        avatar_path = output_dir / "avatars" / f"scene{scene_num}_avatar.mp4"
        if avatar_path.exists():
            jobs.append({
                "kind": "scene_video", "path": avatar_path, "record_id": record_id,
                "scene_num": scene_num, "status_field": None,
            })

    for i, seg in enumerate(segments):
        if i >= len(segment_ids):
            break
        record_id = segment_ids[i]
        img_path = output_dir / "images" / f"scene{seg.scene_number}_seg{seg.segment_index}.png"
        if img_path.exists():
            jobs.append({
                "kind": "segment_image", "path": img_path, "record_id": record_id,
                "seg_label": seg.segment_title, "status_field": "Status_image",
            })
        anim_path = output_dir / "animations" / f"scene{seg.scene_number}_seg{seg.segment_index}_anim.mp4"
        if anim_path.exists():
            jobs.append({
                "kind": "segment_video", "path": anim_path, "record_id": record_id,
                "seg_label": seg.segment_title, "status_field": "Status_video",
            })

    def _do_upload(job: dict, _idx: int) -> bool:
        kind = job["kind"]
        path = job["path"]
        label = job.get("seg_label") or f"scene {job.get('scene_num')}"
        try:
            url = upload_media(wavespeed_key, str(path))
            if kind in ("scene_voice", "scene_video"):
                client.attach_to_scene(job["record_id"], kind, url)
            else:
                client.attach_to_segment(job["record_id"], kind, url)
            if job.get("status_field"):
                if kind.startswith("scene_"):
                    client.update_scene_status(job["record_id"], job["status_field"], "Done")
                else:
                    client.update_segment_status(job["record_id"], job["status_field"], "Done")
            safe_print(f"      ✓ {kind} ({label})")
            return True
        except Exception as e:
            safe_print(f"      ⚠️  {kind} ({label}): {e}")
            return False

    run_parallel(
        jobs,
        _do_upload,
        max_workers=int(os.environ.get("AIRTABLE_UPLOAD_PARALLEL", "4")),
        label="airtable-uploads",
    )
    print(f"   ✓ Asset upload complete")


def _list_voices_and_exit() -> None:
    """Print available ElevenLabs voices and exit."""
    print("\n🎙️  Available ElevenLabs voices:\n")
    try:
        voices = list_voices()
        for v in voices:
            print(f"   {v['voice_id']}  {v['name']:20s}  ({v['category']})")
        print(f"\n   Use --voice-id <id> to select a voice")
    except Exception as e:
        print(f"   Error listing voices: {e}")


def _preflight_check(args) -> None:
    """Validate all prerequisites before spending API credits."""
    errors = []
    needs_script_llm = not (getattr(args, "images_only", False) or getattr(args, "voice_only", False))
    needs_image_llm = not getattr(args, "skip_images", False)

    # Always need OpenAI for script generation + image prompts
    if (needs_script_llm or needs_image_llm) and not os.environ.get("OPENAI_API_KEY"):
        errors.append("OPENAI_API_KEY — needed for script + image prompt generation")

    # Voice generation
    if not getattr(args, "skip_voice", False):
        if not os.environ.get("ELEVENLABS_API_KEY"):
            errors.append("ELEVENLABS_API_KEY — needed for voice generation (or use --skip-voice)")
        if not os.environ.get("ELEVENLABS_VOICE_ID") and not getattr(args, "voice_id", ""):
            errors.append("ELEVENLABS_VOICE_ID or --voice-id — no voice selected")

    # Image generation
    if not getattr(args, "skip_images", False):
        if not os.environ.get("GEMINI_API_KEY"):
            errors.append("GEMINI_API_KEY — needed for image generation (or use --skip-images)")

    # Avatar generation
    if not getattr(args, "skip_avatar", False) and not getattr(args, "skip_voice", False):
        if not os.environ.get("WAVESPEED_API_KEY"):
            errors.append("WAVESPEED_API_KEY — needed for avatar generation (or use --skip-avatar)")
        avatar_img = getattr(args, "avatar_image", "")
        if avatar_img and not Path(avatar_img).exists():
            errors.append(f"Avatar image not found: {avatar_img}")

    # Animation generation
    if not getattr(args, "skip_animation", False):
        if not os.environ.get("KIE_API_KEY"):
            errors.append("KIE_API_KEY — needed for animation generation (or use --skip-animation)")
        if not os.environ.get("WAVESPEED_API_KEY"):
            errors.append("WAVESPEED_API_KEY — also needed to host images for animation generation")

    if getattr(args, "mode", "auto") == "production" and getattr(args, "topic", ""):
        errors.append("Production mode does not allow topic-only runs — provide a PDF or use --mode creative")

    # Airtable
    if not getattr(args, "dry_run", False):
        if not os.environ.get("AIRTABLE_PAT"):
            errors.append("AIRTABLE_PAT — needed for Airtable push (or use --dry-run)")

    # ffmpeg for composition
    if not getattr(args, "skip_compose", False) and not shutil.which("ffmpeg"):
        errors.append("ffmpeg not installed — needed for video composition (brew install ffmpeg or use --skip-compose)")

    if errors:
        print("\n❌ Pre-flight check failed:\n")
        for e in errors:
            print(f"   - {e}")
        print(f"\n   Fix these in .env or add skip flags, then try again.")
        sys.exit(1)

    print("✓ Pre-flight check passed")


def _interactive_source_picker() -> tuple:
    """Ask the user whether to use a PDF or enter a topic string.

    Returns:
        (MedicalContent, source_label) tuple
    """
    print(f"\n📋 How would you like to create your video?")
    print(f"   [1] PDF   — Extract from a medical education PDF")
    print(f"   [2] Topic — Enter a topic and I'll generate the content")
    print(f"   [3] Brief — Enter a full creative brief (topic + style + voice direction)")

    while True:
        choice = input("\n   Choose [1-3]: ").strip()

        if choice == "1":
            pdf_input = input("   PDF path (default: input.pdf): ").strip() or "input.pdf"
            pdf_path = Path(pdf_input)
            if not pdf_path.exists():
                print(f"   Error: File not found: {pdf_path}")
                continue
            print(f"\n📄 Extracting content from {pdf_path.name}...")
            content = extract_pdf(pdf_path)
            return content, f"pdf: {pdf_path.name}"

        elif choice == "2":
            topic = input("   Enter your topic: ").strip()
            if not topic:
                print("   Please enter a topic.")
                continue
            print(f"\n🧠 Generating medical content for: {topic}")
            content = generate_content_from_topic(topic)
            return content, f"topic: {topic}"

        elif choice == "3":
            print("   Enter your creative brief (type 'done' on a new line when finished):")
            lines = []
            while True:
                line = input("   ")
                if line.strip().lower() == "done":
                    break
                lines.append(line)
            brief_text = "\n".join(lines)
            if not brief_text.strip():
                print("   Please enter a brief.")
                continue
            brief = build_brief_from_inputs(request_text=brief_text)
            topic = brief.topic
            if not topic:
                topic = input("   Topic not detected. Enter topic: ").strip()
            print(f"\n🧠 Generating medical content for: {topic}")
            content = generate_content_from_topic(topic)
            return content, f"brief: {topic}"

        else:
            print("   Invalid choice. Enter 1, 2, or 3.")


def _interactive_duration_picker(plan) -> float:
    """Show duration recommendations and let user choose."""
    print(f"\n⏱️  Duration recommendations for \"{plan.topic}\":")
    print(f"   ┌{'─'*58}┐")
    for i, opt in enumerate(plan.duration_options, 1):
        marker = " ◀ recommended" if opt.label == "recommended" else ""
        print(f"   │ [{i}] {opt.label.upper():12s} {opt.minutes:5.1f} min  ({opt.scene_count} scenes){marker:>14s} │")
        print(f"   │     {opt.description[:54]:54s} │")
    print(f"   │ [4] CUSTOM     Enter your own duration                   │")
    print(f"   └{'─'*58}┘")

    while True:
        try:
            choice = input("\n   Choose [1-4] (default: 1): ").strip() or "1"
            if choice == "4":
                mins = input("   Enter duration in minutes: ").strip()
                return float(mins)
            idx = int(choice) - 1
            if 0 <= idx < len(plan.duration_options):
                selected = plan.duration_options[idx]
                print(f"   Selected: {selected.label} — {selected.minutes} min")
                return selected.minutes
        except (ValueError, IndexError):
            pass
        print("   Invalid choice. Try again.")


if __name__ == "__main__":
    main()
