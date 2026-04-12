# Medical Education Longform AI Explainer — Full Python Pipeline

## Context

Safwan has an existing n8n workflow that generates AI explainer videos from medical content, but the script generation is minimal ("Generate a script. do only 2 scenes") and the pipeline is spread across n8n + multiple APIs. He wants to rebuild this as an **all-Python pipeline** that:
- Takes any medical education PDF (like BoardBuddy study diagrams) as input
- Generates a **phenomenal** hybrid narrative + teaching script
- Produces AI-generated medical animations (not just generic b-roll)
- Composites avatar (picture-in-picture) over educational content
- Tracks everything in Airtable
- Eventually edits the final video together (Remotion/ffmpeg — future phase)

The input PDF contains: clinical vignettes, MCQs, pathophysiology explanations, medical diagrams, and educational objectives.

---

## API Keys Required (6 total)

```
ANTHROPIC_API_KEY=           # Claude — script generation + image prompt engineering (you have this)
AIRTABLE_PAT=                # Airtable Personal Access Token
AIRTABLE_BASE_ID=            # appjmdOqi7hTArDN6 (from existing workflow)
ELEVENLABS_API_KEY=          # ElevenLabs TTS (replacing Fish Audio)
GEMINI_API_KEY=              # Already in AWSOM Dashboard .env.local — Nano Banana images (500 FREE/day!) + image analysis
WAVESPEED_API_KEY=           # InfiniteTalk avatar ONLY
KIE_API_KEY=                 # Kling 3.0 video generation ONLY
```

### Optimal Cost Split: 4 providers, minimum spend
| Function | Provider | Model | Price |
|---|---|---|---|
| **Script generation** | **Anthropic** | Claude | Per-token |
| **Image prompt engineering** | **Anthropic** | Claude | Per-token (transforms [VISUAL:] → hyperreal prompts) |
| **Medical images** | **Google (direct)** | Nano Banana Flash via Gemini API | **FREE** (500/day) |
| **Medical images (hero shots)** | **Google (direct)** | Nano Banana Pro via Gemini API | $0.134/image (2K), $0.24 (4K) |
| **Image analysis** | **Google (direct)** | Gemini | Free tier |
| **Medical animation video** | **KIE.ai** | Kling 3.0 image-to-video | ~$0.025/sec (85% cheaper than Wavespeed) |
| **Avatar talking-head** | **Wavespeed** | InfiniteTalk Fast | $0.075/run |
| **Voice/TTS** | **ElevenLabs** | (direct) | ~$0.30/1K chars |

---

## Architecture Overview

### Linear Pipeline (single PDF)
```
input.pdf
    │
    ▼
[1. EXTRACT] ─── PDF text + structure + diagram descriptions
    │
    ▼
[2. ANALYZE] ─── Content scope, teaching plan, scene outline
    │
    ▼
[3. GENERATE SCRIPT] ─── Claude API → full production script with visual directions
    │
    ▼
[4. PUSH TO AIRTABLE] ─── Create Project + Scenes records
    │
    ├──────────────────────────┐
    ▼                          ▼
[5. VOICE]                [6. IMAGES]          ← PARALLEL
ElevenLabs TTS            Claude hyperreal prompt
per scene                 → Gemini Nano Banana
    │                          │
    ▼                          ▼
[7. AVATAR]               [8. ANIMATION]       ← PARALLEL (after deps)
Wavespeed InfiniteTalk    KIE.ai Kling 3.0
(needs voice audio)       (needs image)
    │                          │
    └──────────┬───────────────┘
               ▼
[9. COMPOSITE] ─── Avatar PIP over animations (future: Remotion/ffmpeg)
               ▼
[10. UPDATE AIRTABLE] ─── Log all generated assets
```

### Ralph Loop Architecture (autonomous, parallel, multi-PDF)

Each phase runs as an autonomous **Ralph loop** — polling Airtable for work, processing it, verifying completion, and looping. Multiple loops run in parallel, triggered by status fields.

```
┌─────────────────────────────────────────────────────────┐
│  LOOP 1: Script Generator (polls Project.Status)         │
│  while projects with Status=="Create" exist:             │
│    → Extract PDF → Analyze → Generate Script             │
│    → Push scenes (Status_voice="Create",                 │
│      Status_image="Create")                              │
│    → Mark project "Script_Done"                          │
│    → Verify: all scenes created? word counts valid?      │
│    → Loop                                                │
└────────────────────┬────────────────────────────────────┘
                     │ triggers ↓
    ┌────────────────┴────────────────┐
    ▼                                 ▼
┌──────────────────────┐  ┌──────────────────────┐
│ LOOP 2: Voice        │  │ LOOP 3: Images       │  ← RUN IN PARALLEL
│ polls Status_voice   │  │ polls Status_image   │
│ → ElevenLabs TTS     │  │ → Claude prompt eng  │
│ → Upload audio       │  │ → Gemini Nano Banana │
│ → Status="Done"      │  │ → Status="Done"      │
│ → Verify: audio OK?  │  │ → Verify: image OK?  │
└──────────┬───────────┘  └──────────┬───────────┘
           │ triggers ↓              │ triggers ↓
┌──────────────────────┐  ┌──────────────────────┐
│ LOOP 4: Avatar       │  │ LOOP 5: Animation    │  ← RUN IN PARALLEL
│ polls Status_avatar  │  │ polls Status_anim    │
│ (only if voice Done) │  │ (only if image Done) │
│ → InfiniteTalk       │  │ → Kling 3.0 i2v      │
│ → Poll completion    │  │ → Poll completion    │
│ → Status="Done"      │  │ → Status="Done"      │
│ → Verify: synced?    │  │ → Verify: relevant?  │
└──────────────────────┘  └──────────────────────┘
```

**Key Ralph principles applied:**
- Each loop has **explicit completion criteria** (not just "did it run" but "is the output valid")
- Each loop **self-verifies** before marking done (audio duration matches script? image is medical content?)
- Loops are **independent processes** — can run on different terminals/machines
- Airtable status fields act as the **message queue** between loops
- Upload 10 PDFs → all 10 process through the pipeline autonomously
- `run.py` can launch all loops with: `python run.py --loop all` or individual: `python run.py --loop voice`

---

## File Structure

```
Medical Education longform automation/
├── .env                          # API keys (gitignored)
├── .env.example                  # Template
├── .gitignore
├── pyproject.toml
├── CLAUDE.md                     # Project-specific instructions
├── run.py                        # CLI entry: python run.py input.pdf
│
├── tools/
│   ├── __init__.py
│   ├── models.py                 # Pydantic data models for everything
│   │
│   ├── extract.py                # PDF parsing + section classification
│   │                             # Uses PyMuPDF (fitz) for text extraction
│   │                             # Regex-based section detection (BoardBuddy format)
│   │                             # Outputs: MedicalContent dataclass
│   │
│   ├── analyze.py                # Scope analysis + teaching plan generation
│   │                             # Determines: video length, scene count, narrative arc
│   │                             # Deterministic Python logic, no LLM call
│   │                             # Outputs: TeachingPlan with scene briefs
│   │
│   ├── generate_script.py        # Claude API script generation (the core)
│   │                             # Two-pass: creative generation → quality review
│   │                             # Strips production tags for TTS-clean version
│   │                             # Outputs: ProductionScript with scenes
│   │
│   ├── voice.py                  # ElevenLabs TTS generation per scene
│   │                             # Sends clean script text → receives MP3
│   │                             # Outputs: audio file paths
│   │
│   ├── avatar.py                 # KIE.ai InfiniteTalk avatar video
│   │                             # Sends voice audio + reference image → talking head video
│   │                             # Handles polling for completion
│   │                             # Outputs: video file paths
│   │
│   ├── animations.py             # AI medical animation generation via KIE.ai
│   │                             # Uses [VISUAL:] tags from script as prompts
│   │                             # Image generation (Flux Kontext) → video generation (Kling 3.0)
│   │                             # Outputs: animation video file paths per scene
│   │
│   ├── airtable_client.py        # Airtable CRUD for Project/Scenes/Segments tables
│   │                             # Create project, push scenes, update statuses, log assets
│   │
│   └── quality.py                # Script quality validation
│                                 # Word count checks, AI slop detection, completeness
│
├── data/
│   └── prompts/
│       └── system_prompt.txt     # Master script generation prompt (versioned)
│
├── workflows/
│   └── pdf_to_video.md           # SOP for the full pipeline
│
├── output/                       # Generated assets (gitignored)
│   └── {project_name}/
│       ├── script.json           # Full production script
│       ├── voices/               # MP3 files per scene
│       ├── avatars/              # Avatar videos per scene
│       └── animations/           # Medical animation videos per scene
│
└── input.pdf                     # Example input (already exists)
```

---

## Build Approach

Build normally in a Claude Code session (phases 1 → 2 → 3). Ralph loop is **not needed for building** — it's overkill for ~11 items when you're actively collaborating.

**Where Ralph IS useful:** once the pipeline is built, the *runtime* can use Ralph-style loops to batch-process multiple PDFs autonomously (see Ralph Loop Architecture above).

---

## Phase 1: Foundation + Script Generation (Build First)

### PRD Items — Phase 1

Each item has a **verification command** that must pass before marking complete.

- [ ] **1.1 Project setup**
  - Create `.env.example`, `.gitignore`, `pyproject.toml`, `CLAUDE.md`
  - Dependencies: `anthropic`, `pymupdf` (fitz), `pyairtable`, `httpx`, `pydantic`, `python-dotenv`, `elevenlabs`
  - Init git repo, create venv, install deps
  - **Verify:** `python -c "import anthropic, fitz, pyairtable, httpx, pydantic, dotenv; print('OK')"`

- [ ] **1.2 Data models** (`tools/models.py`)
  - `MedicalContent` — extracted PDF content (vignette, MCQ, pathophysiology, etc.)
  - `TeachingPlan` / `SceneBrief` — structural outline
  - `ProductionScene` — scene with narration + visual directions + avatar cues
  - `ProductionScript` — the complete script
  - **Verify:** `python -c "from tools.models import MedicalContent, TeachingPlan, ProductionScene, ProductionScript; print('OK')"`

- [ ] **1.3 PDF extraction** (`tools/extract.py`)
  - PyMuPDF extracts text by page
  - Regex classifies sections: Question, Answer Choices, Correct Answer, Pathophysiology, Key Info, Why, Explanation, Educational Objective, Image Prompt
  - Returns structured `MedicalContent`
  - **Verify:** `python -c "from tools.extract import extract_pdf; result = extract_pdf('input.pdf'); assert result.topic; assert result.clinical_vignette; assert result.pathophysiology; assert result.educational_objective; assert len(result.answer_choices) >= 4; print('OK:', result.topic)"`

- [ ] **1.4 Content analysis** (`tools/analyze.py`)
  - Count concepts, differential diagnoses, pathophysiology chain steps
  - Calculate recommended length: base 3 min + depth adjustments
  - Generate `TeachingPlan` with ordered scene briefs:
    - Scene 1: Patient hook (from vignette)
    - Scene 2: Clinical puzzle (the question)
    - Scene 3+: Mechanism teaching (pathophysiology, step by step)
    - Penultimate: Differentials (why not the other answers)
    - Final: Clinical pearl + educational objective
  - **Verify:** `python -c "from tools.extract import extract_pdf; from tools.analyze import analyze_content; content = extract_pdf('input.pdf'); plan = analyze_content(content); assert plan.recommended_minutes > 0; assert len(plan.scenes) >= 3; assert plan.scenes[0].purpose == 'hook'; print('OK:', len(plan.scenes), 'scenes,', plan.recommended_minutes, 'min')"`

- [ ] **1.5 Script generation** (`tools/generate_script.py`)
  - **Pass 1:** Claude API call with extracted content + teaching plan → full production script
  - **Pass 2:** Claude reviews its own script for quality, medical accuracy, engagement
  - Production tags embedded inline: `[MODE:]`, `[VISUAL:]`, `[TEXT:]`, `[AVATAR:]`, `[PACE:]`
  - `[MODE: avatar|animation|overlay]` controls what's on screen at each moment
  - `script` field = tags stripped (for TTS), `script_full` = tags intact (for animation prompts + composition)
  - **Master prompt** stored in `data/prompts/system_prompt.txt`:
    - Role: Senior resident teaching a junior, studied Osmosis/Ninja Nerd style
    - Hybrid narrative + teaching structure
    - Rhetorical questions for micro-suspense
    - Analogies for complex mechanisms
    - Blocklist: "Let's dive in", "In this video", "without further ado", etc.
  - **Verify:** `python -c "from tools.extract import extract_pdf; from tools.analyze import analyze_content; from tools.generate_script import generate_script; content = extract_pdf('input.pdf'); plan = analyze_content(content); script = generate_script(content, plan); assert len(script.scenes) >= 3; assert all('[MODE:' in s.script_full for s in script.scenes); assert all(s.word_count > 0 for s in script.scenes); print('OK:', len(script.scenes), 'scenes,', script.total_minutes, 'min')"`
  - **Requires:** `ANTHROPIC_API_KEY` in `.env`

- [ ] **1.6 Quality validation** (`tools/quality.py`)
  - Word count per scene matches duration (140-160 WPM)
  - All scenes have `[VISUAL:]` directions
  - All scenes have `[MODE:]` tags
  - No scene > 5 minutes
  - Educational objective addressed somewhere in the script
  - AI slop phrase blocklist detection
  - **Verify:** `python -c "from tools.quality import validate_script; from tools.models import ProductionScript; import json; script = ProductionScript.model_validate_json(open('output/test/script.json').read()); issues = validate_script(script); print('Issues:', issues); assert len(issues) == 0"`

- [ ] **1.7 Airtable integration** (`tools/airtable_client.py`)
  - CRUD wrapper for Project, Scenes, Segments tables
  - Create project + push scenes with `Status_voice = "Create"`, `Status_image = "Create"`
  - Field mapping:
    - `scene` → `"N - Title"` format
    - `script` → clean narration (TTS-ready)
    - `speech_prompt` → avatar delivery cues
    - `estimate_mins` → scene duration
  - **Verify:** `python -c "from tools.airtable_client import AirtableClient; client = AirtableClient(); tables = client.list_tables(); print('Tables:', tables)"`
  - **Requires:** `AIRTABLE_PAT` and `AIRTABLE_BASE_ID` in `.env`

- [ ] **1.8 CLI entry point** (`run.py`)
  - `python run.py input.pdf` — interactive mode:
    1. Extracts PDF, analyzes content
    2. Shows **smart duration recommendation** with reasoning:
       - Recommended (covers all content well)
       - Minimum (core concept only)
       - Deep dive (all differentials + clinical pearls)
       - Custom (user enters minutes)
    3. User picks duration → script generates to match
    4. Pushes to Airtable
  - `python run.py input.pdf --dry-run` — generates script locally without pushing
  - `python run.py input.pdf --duration 8` — skip the prompt, use 8 minutes
  - Saves full output to `output/{project_name}/script.json`
  - **Verify:** `python run.py input.pdf --dry-run` exits 0 and creates `output/*/script.json`

---

## Phase 2: Voice + Avatar Generation (Build Second)

### PRD Items — Phase 2

- [ ] **2.1 Voice generation** (`tools/voice.py`)
  - ElevenLabs TTS API: send clean script text per scene
  - Configure voice ID, stability, similarity boost, style settings
  - Save MP3 files to `output/{project}/voices/`
  - Upload to Airtable scene records
  - **Verify:** `python -c "from tools.voice import generate_voice; audio = generate_voice('This is a test sentence.', output_path='output/test/test.mp3'); import os; assert os.path.exists('output/test/test.mp3'); assert os.path.getsize('output/test/test.mp3') > 1000; print('OK')"`
  - **Requires:** `ELEVENLABS_API_KEY` in `.env`

- [ ] **2.2 Avatar video** (`tools/avatar.py`)
  - Wavespeed InfiniteTalk API: send voice audio + reference image
  - Poll for completion (with exponential backoff)
  - Save avatar videos to `output/{project}/avatars/`
  - Update Airtable
  - **Verify:** `python -c "from tools.avatar import generate_avatar; video = generate_avatar('output/test/test.mp3', 'path/to/reference.jpg', output_path='output/test/test_avatar.mp4'); import os; assert os.path.exists(video); print('OK:', video)"`
  - **Requires:** `WAVESPEED_API_KEY` in `.env`

---

## Phase 3: Medical Animations (Build Third)

### PRD Items — Phase 3

- [ ] **3.1 Image prompt engineering** (`tools/image_prompts.py`)
  - Claude API + Hyperreal Med Image system prompt transforms `[VISUAL:]` tags into Nano Banana-ready prompts
  - System prompt stored in `data/prompts/image_system_prompt.txt`
  - **Verify:** `python -c "from tools.image_prompts import generate_image_prompt; prompt = generate_image_prompt('Cross-section of heart showing blood backing up from LV through LA into pulmonary veins', scene_context='pathophysiology'); assert 'hyperrealistic' in prompt.lower() or '16K' in prompt; assert len(prompt) > 200; print('OK:', prompt[:100])"`

- [ ] **3.2 Image generation** (`tools/animations.py`)
  - Generate key frame images via **Gemini API — Nano Banana Flash** (FREE, 500/day) or **Nano Banana Pro** ($0.134 for hero shots)
  - Save to `output/{project}/animations/`
  - **Verify:** `python -c "from tools.animations import generate_image; path = generate_image('A 16K hyperrealistic rendering of a cross-section of the human heart...', output_path='output/test/test_frame.png'); import os; assert os.path.exists(path); print('OK:', path)"`
  - **Requires:** `GEMINI_API_KEY` in `.env`

- [ ] **3.3 Video animation** (`tools/animations.py`)
  - KIE.ai Kling 3.0 image-to-video from key frame
  - Poll for completion
  - Save to `output/{project}/animations/`
  - Update Airtable segments
  - **Verify:** `python -c "from tools.animations import generate_video_from_image; path = generate_video_from_image('output/test/test_frame.png', prompt='Blood flows backward through the heart chambers', output_path='output/test/test_anim.mp4'); import os; assert os.path.exists(path); print('OK:', path)"`
  - **Requires:** `KIE_API_KEY` in `.env`

### Hyperreal Med Image Prompt System (adapted from custom GPT)

Stored in `data/prompts/image_system_prompt.txt`. This is a system prompt used by Claude to transform raw `[VISUAL:]` scene descriptions into hyperrealistic, medically accurate image generation prompts for Nano Banana Pro.

**Input:** Scene script context + `[VISUAL:]` tag content
**Output:** A single detailed image generation prompt

```
ROLE: Hyperreal Medical Image Prompt Engineer

You receive a scene script and its [VISUAL:] description from a medical education
video. Your job: transform it into a single, hyperdetailed image generation prompt
optimized for Nano Banana Pro.

MEDICAL CONTEXT INTELLIGENCE:
- Recognize and visualize anatomy, physiology, pathology, microbiology, molecular biology
- Automatically determine appropriate magnification and environment:
  - Neuron → synaptic cleft environment
  - Herpesvirus → mucosal epithelium
  - RBC → capillary flow
  - Heart failure → cross-section with pressure gradients
- Disregard real-world optical scale limits to visualize microscopic or internal
  subjects with full photoreal accuracy

CINEMATIC MATERIAL REALISM:
- HDRI lighting, volumetric diffusion, subsurface scattering
- Micro-surface fidelity, sub-millimeter detail
- Film-grade physical properties rendered in Octane style

VISUALIZATION MODES (choose based on context):
- clinical: lifelike medical realism
- illustrative: simplified clarity for teaching
- pathophysiology: visualizes internal biological processes
- comparison: healthy vs diseased tissue
- modality: MRI, CT, X-ray, Ultrasound, PET, Histology, SEM/TEM style

PROMPT TEMPLATE:
"Create a 16K ultra hyperrealistic cinematic rendering of [SUBJECT] in its most
contextually accurate environment ([ENVIRONMENT]), captured with Canon EOS R5
using 85mm f/1.4 lens, featuring HDRI natural light, volumetric diffusion,
subsurface scattering, ultra-micro surface texture fidelity, and sub-millimeter
detail. Render with Octane. Disregard real-world optical scale limits to visualize
with full photoreal accuracy. Maintain cinematic lighting continuity and medical
realism. Description: [PHYSICAL_DESCRIPTION]."

RULES:
- Output ONLY the image prompt, nothing else
- Fill in [SUBJECT], [ENVIRONMENT], and [PHYSICAL_DESCRIPTION] from the scene context
- Include specific anatomical/pathological details from the script
- Maintain lighting continuity across scenes (consistent HDRI direction)
- For pathophysiology scenes: show the mechanism in action (e.g., pressure backing
  up, fluid accumulating, vessels dilating)
```

---

## Phase 4: Video Composition (Future — Discuss Later)

### Visual Modes (the final video cuts between these)

The script controls which visual mode is active at each moment via `[MODE:]` tags:

| Mode | What's on screen | When to use |
|---|---|---|
| `[MODE: avatar]` | **Full-screen talking head** | Hook/intro, emotional moments, direct teaching, transitions |
| `[MODE: animation]` | **Full-screen medical animation** | Detailed mechanism shots, diagrams, the visual IS the lesson |
| `[MODE: overlay]` | **Avatar PIP (corner bubble) over animation** | Guided walkthroughs — narrator explains what you're seeing |

Example flow for a scene about pressure backing up:
```
[MODE: avatar]
"So here's the million dollar question — if the left ventricle is the problem, 
why are the lungs filling with fluid?"

[MODE: animation]
[VISUAL: Cross-section of heart showing blood backing up from thickened LV 
through dilated LA into engorged pulmonary veins. Pressure arrows animate 
backward. Veins visibly swell.]

[MODE: overlay]
"See this right here? The pressure has nowhere to go. It backs up through 
the left atrium..."
[TEXT: "Backward Pressure Transmission" appears bottom-third]

[MODE: avatar]
[AVATAR: Slight nod, reassuring expression]
"And that's exactly why this patient can't lie flat at night."
```

This means:
- **Avatar clips and animation clips can be different lengths** — the video is driven by the script timing, not fixed clip durations
- **The composition step stitches them together** based on `[MODE:]` tags + audio timing
- **Total video length is dynamic** — some moments need 10 seconds of animation, others need 3 seconds of avatar

### Step 12: Composite editing
- Parse `[MODE:]` tags from script to create an **edit decision list (EDL)**
- Align avatar video, animation clips, and audio on a timeline
- Three composition modes:
  - `avatar` → full-screen avatar video
  - `animation` → full-screen animation clip
  - `overlay` → animation as background, avatar in PIP bubble (corner/side, with rounded mask + subtle shadow)
- Text overlays from `[TEXT:]` tags (positioned based on mode — bottom-third for overlay, center for animation)
- Pacing from `[PACE:]` tags (controls cuts, holds, transitions)
- Options: Remotion (React-based, programmable) or ffmpeg (simpler, Python-native via moviepy)
- This is a significant piece of work — we'll design it after Phases 1-3 are working

---

## Airtable Table Schema (I'll Create These)

### Project Table
| Field | Type | Purpose |
|---|---|---|
| Project Name | Single line text | Primary key |
| Status | Single select | Create / Processing / Done / Error |
| INPUT_Request | Long text | Topic/description from PDF |
| INPUT_voice_id | Single line text | ElevenLabs voice ID |
| INPUT_Image_1 | Attachment | Avatar reference image |
| INPUT_Image_2 | Attachment | Style reference for animations |
| Source_PDF | Attachment | Original input PDF |
| Total_Minutes | Number | Calculated video length |
| aspect_ratio | Single line text | 16:9 default |

### Scenes Table
| Field | Type | Purpose |
|---|---|---|
| id | Auto number | Primary key |
| Project Name | Single line text | Links to project |
| scene | Single line text | "1 - Scene Title" |
| estimate_mins | Number | Scene duration |
| script | Long text | Clean narration (TTS-ready) |
| script_full | Long text | Full script with production tags |
| speech_prompt | Long text | Avatar delivery cues |
| Status_voice | Single select | Create / Done / Skip |
| scene_voice | Attachment | Generated MP3 |
| Status_video | Single select | Create / Done / Skip |
| scene_video | Attachment | Generated avatar video |
| Status_animation | Single select | Create / Done / Skip |
| scene_animation | Attachment | Generated medical animation |
| visual_summary | Long text | Description of scene visuals |

### Segments Table (for individual visual clips within a scene)
| Field | Type | Purpose |
|---|---|---|
| id | Auto number | Primary key |
| Project Name | Single line text | Links to project |
| Scene Name | Single line text | Links to scene |
| segment | Single line text | "Segment 1 - Visual Title" |
| image_prompt | Long text | Image generation prompt |
| video_prompt | Long text | Video generation prompt |
| Status_image | Single select | Create / Done / Skip |
| segment_image | Attachment | Generated image |
| Status_video | Single select | Create / Done / Skip |
| segment_video | Attachment | Generated video clip |

---

## Verification Plan

### After Phase 1:
1. Run `python run.py input.pdf --dry-run` with the sample BoardBuddy PDF
2. Verify extracted content matches PDF sections (all fields populated)
3. Read the generated script — does it feel like Osmosis/Ninja Nerd quality?
4. Check Airtable for correctly created Project + Scenes records
5. Verify scene format matches `"N - Title"` pattern
6. Confirm word counts align with durations (140-160 WPM)

### After Phase 2:
7. Listen to generated voice audio — natural? Correct pacing?
8. Watch avatar videos — synced to audio? Natural movement?
9. Verify Airtable records updated with asset URLs

### After Phase 3:
10. Watch medical animations — relevant to scene content? Medically sensible?
11. Check animation prompts derived correctly from `[VISUAL:]` tags

---

## Key Design Decisions

- **Python-only, no n8n** — simpler stack, full control, one codebase
- **Airtable stays** — visual tracking, already has the schema pattern
- **Two-pass script generation** — creative pass then quality review, prevents AI slop
- **Deterministic teaching plan before LLM** — structure prevents shapeless output
- **Production tags inline** — `[MODE:]`, `[VISUAL:]`, `[TEXT:]`, `[AVATAR:]`, `[PACE:]` embedded in script, stripped for TTS, kept for animation prompts + video composition. `[MODE:]` drives the edit — avatar full screen, animation full screen, or avatar PIP over animation.
- **Gemini API direct for images** — Nano Banana Flash is FREE (500/day), Pro at $0.134 for hero shots. Uses existing Gemini key from AWSOM Dashboard.
- **Wavespeed for avatar only** — InfiniteTalk Fast ($0.075/run) is cheapest for talking-head video
- **KIE.ai for Kling video only** — 85% cheaper than Wavespeed for video generation ($0.025/sec vs $0.168/sec)
- **ElevenLabs over Fish Audio** — better voice quality, more natural prosody, per user preference
- **Hyperreal Med Image system** — Claude transforms raw [VISUAL:] tags into 16K hyperrealistic medical image prompts (adapted from custom GPT), then Nano Banana Pro generates them
