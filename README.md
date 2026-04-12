# Medical Education Longform AI Explainer Pipeline

An all-Python pipeline that transforms medical education PDFs into production-ready AI explainer videos with generated visuals, natural voice narration, and a lip-synced avatar presenter.

**PDF in, video out.** One command runs the full pipeline — from content extraction through script generation, image creation, voice synthesis, avatar animation, and final video composition.

---

## How It Works

```
input.pdf
  |
  v
[1] Extract content (PyMuPDF) ──> structured medical content
  |
  v
[2] Analyze complexity ──> teaching plan with duration options
  |
  v
[3] Generate script (GPT-4o) ──> production-tagged scenes
  |
  v
[4] Break into segments ──> ~5-sec visual beats with image/video prompts
  |
  |── [5a] Generate images (Gemini) ──> hyperrealistic medical PNGs
  |── [5b] Generate voice (ElevenLabs) ──> per-scene MP3 narration
  |── [5c] Character sheet (Gemini) ──> consistent character reference
  |
  v
  |── [6a] Avatar video (Wavespeed) ──> lip-synced talking head
  |── [6b] Animation clips (KIE.ai/Kling) ──> animated medical visuals
  |
  v
[7] Compose final video (ffmpeg or Remotion) ──> 1920x1080 MP4
  |
  v
[8] Push to Airtable ──> track status + assets
```

### Pipeline Stages in Detail

#### 1. PDF Extraction (`tools/extract.py`)
Parses BoardBuddy-format medical PDFs using PyMuPDF. Extracts topic, clinical vignette, MCQ questions, pathophysiology, differential diagnoses, and educational objectives into a structured `MedicalContent` object.

#### 2. Content Analysis (`tools/analyze.py`)
Purely deterministic (no LLM). Counts concepts, differentials, and mechanism steps. Calculates a complexity score (1-5) and generates three duration options:
- **Recommended** — balanced coverage of all concepts
- **Minimum** — core mechanism only
- **Deep Dive** — full differentials + clinical pearls

Outputs a `TeachingPlan` with scene briefs labeled by purpose (hook, question, mechanism, differential, takeaway).

#### 3. Script Generation (`tools/generate_script.py`)
Two-pass generation using GPT-4o:
1. **Creative pass** — generates full scenes from the teaching plan
2. **Quality review** — refines for accuracy, pacing, and anti-slop

Each scene is tagged with inline production directions:
```
[MODE: animation] The mitral valve snaps shut as pressure builds...
[VISUAL: Cross-section of left ventricle, pressure gradient overlay]
[TEXT: Systolic Dysfunction]
[AVATAR: gestures to diagram] And this is where things go wrong.
[PACE: slow] Let that sink in for a moment.
```

- `script` field = tags stripped (clean text for TTS)
- `script_full` field = tags intact (for video composition)
- Word count matched to target duration at 150 WPM

#### 4. Visual Segmentation (`tools/generate_segments.py`)
Breaks each scene into ~5-second segments. Each segment gets:
- An **intent** label (clinical_scene, mechanism, anatomy, molecular, etc.)
- A **hyperrealistic image prompt** engineered for medical accuracy
- A **video motion prompt** for Kling animation

#### 5. Asset Generation (parallel)

| Asset | Tool | Provider | What it does |
|-------|------|----------|-------------|
| Medical images | `generate_images.py` | Gemini 3 Pro | 1024x1024 hyperrealistic PNGs with QA validation |
| Voice narration | `generate_voice.py` | ElevenLabs | Natural TTS per scene, drives timeline |
| Character sheet | `character_sheet.py` | GPT-4o + Gemini | Extracts character spec from vignette, generates reference for consistency |

#### 6. Video Generation (parallel)

| Asset | Tool | Provider | What it does |
|-------|------|----------|-------------|
| Avatar clips | `avatar.py` | Wavespeed InfiniteTalk | Voice + reference face = lip-synced talking head |
| Animation clips | `animations.py` | KIE.ai Kling 3.0 | Image + motion prompt = 5-sec animated medical visual |

#### 7. Video Composition
Two options:
- **ffmpeg** (`tools/compose.py`) — Python-native, parses `[MODE:]` tags to switch between avatar full-screen, animation full-screen, and avatar picture-in-picture over animation
- **Remotion** (`tools/compose_remotion.py`) — React-based, frame-accurate with fade transitions and text animations

#### 8. Airtable Sync (`tools/airtable_client.py`)
Pushes project metadata, scenes, and segments to Airtable with status tracking fields. Each asset (voice, image, animation) has its own status column for batch processing workflows.

---

## Quick Start

### Prerequisites
- Python 3.11+
- FFmpeg (`brew install ffmpeg` on macOS)
- Node.js 18+ (only if using Remotion composition)

### Setup

```bash
# Clone the repo
git clone https://github.com/sarkersafwan/Med-Ed-Explainer-Beta.git
cd Med-Ed-Explainer-Beta

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .

# Copy the environment template and fill in your API keys
cp .env.example .env
# Edit .env with your keys (see API Keys section below)
```

### Avatar Reference Image

The pipeline needs a **face reference image** to generate the lip-synced avatar presenter. This is the "talking head" that appears in `[MODE: avatar]` and `[MODE: overlay]` scenes.

Provide any front-facing portrait photo (PNG or JPG):

```bash
# Pass it as a flag
python run.py input.pdf --avatar-image path/to/face.png

# Or place it at the project root as MedVidSpeaker.png
# and the Streamlit UI / GUI will pick it up automatically
```

Without an avatar image, the pipeline skips avatar generation and only produces animation-based scenes.

### Run

```bash
# Full pipeline from a PDF
python run.py input.pdf

# From a topic (no PDF needed)
python run.py --topic "Heart Failure with Preserved Ejection Fraction"

# Dry run — generate script only, no API calls for media
python run.py input.pdf --dry-run

# Set duration directly (skip interactive prompt)
python run.py input.pdf --duration 8
```

### Streamlit UI

```bash
streamlit run app.py
```

Web-based interface for uploading PDFs, selecting duration, and monitoring pipeline progress.

---

## API Keys

You need accounts with these providers. Copy `.env.example` to `.env` and fill in your keys:

| Variable | Provider | What it's for | Free tier? |
|----------|----------|---------------|------------|
| `OPENAI_API_KEY` | [OpenAI](https://platform.openai.com/) | Script generation, review, image prompt engineering | No |
| `GEMINI_API_KEY` | [Google AI Studio](https://aistudio.google.com/) | Medical image generation | Yes (500 images/day) |
| `ELEVENLABS_API_KEY` | [ElevenLabs](https://elevenlabs.io/) | Voice narration (TTS) | Limited free tier |
| `ELEVENLABS_VOICE_ID` | ElevenLabs | Which voice to use | — |
| `WAVESPEED_API_KEY` | [Wavespeed](https://wavespeed.ai/) | Avatar lip-sync video | No |
| `KIE_API_KEY` | [KIE.ai](https://kie.ai/) | Kling 3.0 animation clips | No |
| `AIRTABLE_PAT` | [Airtable](https://airtable.com/) | Project/asset tracking | Yes |
| `AIRTABLE_BASE_ID` | Airtable | Your base ID | — |

### Optional variables

```env
OPENAI_MODEL=gpt-4o                    # LLM model (default: gpt-4o)
GEMINI_IMAGE_MODEL=gemini-3-pro-image-preview  # Image model
PIPELINE_MODE=auto                      # auto, production, or creative
```

---

## CLI Reference

```bash
# === Full pipeline ===
python run.py input.pdf                     # Interactive duration selection
python run.py input.pdf --duration 5        # 5-minute target
python run.py --topic "Muscle Contraction"  # Topic-only (no PDF)

# === Partial pipeline ===
python run.py input.pdf --dry-run           # Script only, no media generation
python run.py input.pdf --images-only       # Images from existing script.json
python run.py input.pdf --voice-only        # Voice from existing script.json

# === Skip specific stages ===
python run.py input.pdf --skip-images
python run.py input.pdf --skip-voice
python run.py input.pdf --skip-avatar
python run.py input.pdf --skip-animation

# === Overrides ===
python run.py input.pdf --voice-id JBFqnCBsd6RMkjVDRZzb
python run.py input.pdf --avatar-image face.png

# === Airtable setup ===
python setup_airtable.py          # Validate schema
python setup_airtable.py --fix    # Auto-fix table IDs
```

---

## Project Structure

```
medical-edu-explainer/
├── run.py                    # Main CLI entry point
├── app.py                    # Streamlit web UI
├── setup_airtable.py         # Airtable schema validator
├── pyproject.toml            # Dependencies
├── .env.example              # Environment template
│
├── tools/                    # Pipeline modules (WAT framework)
│   ├── extract.py            #   PDF parsing (PyMuPDF)
│   ├── analyze.py            #   Deterministic teaching plan
│   ├── generate_script.py    #   GPT-4o script generation
│   ├── generate_segments.py  #   Visual segment breakdown
│   ├── generate_images.py    #   Gemini image generation
│   ├── generate_voice.py     #   ElevenLabs TTS
│   ├── avatar.py             #   Wavespeed lip-sync avatar
│   ├── animations.py         #   KIE.ai Kling animation
│   ├── character_sheet.py    #   Character consistency
│   ├── compose.py            #   ffmpeg video composition
│   ├── compose_remotion.py   #   Remotion video composition
│   ├── creative_brief.py     #   Airtable creative direction
│   ├── review.py             #   Source grounding artifacts
│   ├── quality.py            #   Script validation
│   ├── alignment.py          #   Segment timing sync
│   ├── parallel.py           #   Parallel execution utils
│   ├── project_store.py      #   Run/project filesystem
│   ├── provider.py           #   LLM provider abstraction
│   ├── airtable_client.py    #   Airtable CRUD
│   └── models.py             #   Pydantic data models
│
├── data/
│   └── prompts/              # Versioned system prompts
│       ├── system_prompt.txt  #   Script generation prompt
│       ├── image_prompt.txt   #   Image engineering prompt
│       └── video_prompt.txt   #   Video motion prompt
│
├── remotion/                 # React/Remotion video editor (optional)
│   ├── src/
│   ├── package.json
│   └── tsconfig.json
│
├── tests/                    # Test suite
│
└── output/                   # Generated assets (gitignored)
    └── {project}/
        └── runs/{run_id}/
            ├── script.json
            ├── segments.json
            ├── voice/         # Scene MP3 files
            ├── images/        # Segment PNGs
            ├── avatars/       # Avatar MP4s
            ├── animations/    # Animation MP4s
            └── final_video.mp4
```

---

## Production Tags

The script generation system uses inline tags to direct video composition:

| Tag | Purpose | Example |
|-----|---------|---------|
| `[MODE: avatar\|animation\|overlay]` | Camera mode for the scene | `[MODE: overlay]` = avatar PIP over animation |
| `[VISUAL: ...]` | What the viewer sees | `[VISUAL: CT scan revealing pleural effusion]` |
| `[TEXT: ...]` | On-screen text overlay | `[TEXT: Diastolic Dysfunction]` |
| `[AVATAR: ...]` | Avatar delivery direction | `[AVATAR: leans in, concerned expression]` |
| `[PACE: slow\|normal\|fast]` | Narration pacing | `[PACE: slow]` for emphasis |

**Rules enforced:**
- First scene never opens in avatar mode (visual hook first)
- At least 50% of scenes use animation or overlay mode
- Every scene has `[VISUAL:]` and `[MODE:]` tags

---

## Airtable Schema

Run `python setup_airtable.py` to validate your base has the right tables and fields.

### Tables

**Projects** — one row per video project
- Project Name, Status, INPUT_Request, Source_PDF, Total_Minutes

**Scenes** — one row per script scene
- Project Name, scene label, script (clean), script_full (tagged), Status_voice, Status_animation

**Segments** — one row per ~5-sec visual beat
- Project Name, Scene Name, image_prompt, video_prompt, Status_image, Status_video

Status fields use: `Create` | `Done` | `Skip` | `Error`

---

## Architecture Notes

**WAT Framework** — Workflows, Agents, Tools. Probabilistic AI (LLM reasoning) is separated from deterministic Python (execution). Each `tools/` module is a standalone, testable unit.

**Two-pass script generation** — Creative generation followed by quality review catches AI slop and ensures medical accuracy.

**Deterministic teaching plan** — Content analysis constrains the LLM before it writes, preventing shapeless output.

**Character consistency** — A character spec is derived once from the clinical vignette and passed as a reference image to all human-focused segments.

**Network resilience** — All external API calls use exponential backoff retry. A single network hiccup doesn't lose generated assets.

**Run versioning** — Each pipeline execution creates a timestamped run directory. The project keeps the 2 most recent runs by default.

---

## Example Output

```
$ python run.py input.pdf

Extracting content from input.pdf...
Topic: Heart Failure with Preserved Ejection Fraction (HFpEF)
Complexity: 3/5 | Concepts: 4 | Differentials: 3

Duration options:
  [1] Recommended (5.2 min) — all concepts + differentials
  [2] Minimum (3.0 min) — core mechanism only
  [3] Deep Dive (7.5 min) — differentials + clinical pearls
  [4] Custom

Pick duration [1]: 1

Generating script (2-pass)... done (5 scenes, 780 words)
Creating Airtable project... done
Generating segments... 18 segments across 5 scenes
Generating voice... 5/5 scenes complete
Generating character sheet... done
Generating images... 18/18 segments complete
Generating avatar videos... 5/5 scenes complete
Generating animation clips... 18/18 segments complete
Composing final video... done

Output: output/hfpef/runs/20240407_181500/final_video.mp4
```

---

## Cost Estimate (per video)

| Stage | Provider | Approximate cost |
|-------|----------|-----------------|
| Script + prompts | OpenAI GPT-4o | ~$0.10-0.30 |
| Medical images | Gemini Flash | FREE (500/day) |
| Voice narration | ElevenLabs | ~$0.15-0.50 |
| Avatar videos | Wavespeed | ~$0.40-0.75 |
| Animation clips | KIE.ai Kling | ~$0.50-1.00 |
| **Total** | | **~$1.15-2.55 per video** |

Costs vary with video length and number of scenes/segments.

---

## License

MIT
