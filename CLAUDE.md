# Medical Education Longform AI Explainer Pipeline

## What This Is
All-Python pipeline that transforms medical education PDFs into production-ready AI explainer video scripts with full visual direction.

## Architecture
- `tools/` — deterministic Python modules (WAT framework)
- `data/prompts/` — versioned system prompts for script + image generation
- `output/` — generated assets per project (gitignored)
- Airtable tracks pipeline state (base: appjmdOqi7hTArDN6)

## API Providers
- **OpenAI-compatible text/vision model** — script generation, script review, and image prompt engineering
- **ElevenLabs** — voice/TTS
- **Wavespeed** — InfiniteTalk avatar video only
- **KIE.ai** — Kling 3.0 video generation only
- **Gemini** — Nano Banana image generation (500 free/day)

## Key Conventions
- Production tags in scripts: `[MODE:]`, `[VISUAL:]`, `[TEXT:]`, `[AVATAR:]`, `[PACE:]`
- `script` field = tags stripped (TTS-ready), `script_full` = tags intact
- Secrets in `.env`, never hardcoded
- Test against `input.pdf` (BoardBuddy format)
- `output/<project>/runs/<run_id>/` is the source of truth for generated runs
- `output/<project>/character/latest/` mirrors the latest character sheet for the project
