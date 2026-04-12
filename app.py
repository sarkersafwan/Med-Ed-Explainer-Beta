"""Web GUI for the Medical Education Video Pipeline.

Cinematic dark-theme Streamlit interface — prestige medical drama aesthetic.
Run with: streamlit run app.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from tools.project_store import get_latest_run_id, list_project_runs

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_AVATAR = PROJECT_ROOT / "MedVidSpeaker.png"

st.set_page_config(
    page_title="Medical Longform — AI Explainer Pipeline",
    page_icon="◐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================================================================
# CINEMATIC THEME — prestige medical drama
# =============================================================================

CINEMATIC_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,500;1,9..144,300;1,9..144,500&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg-0: #0B0D10;
  --bg-1: #13161B;
  --bg-2: #1B2028;
  --bg-3: #242A34;
  --border: #2A2F38;
  --border-bright: #3A404B;
  --text-0: #E8E6E1;
  --text-1: #9BA3AD;
  --text-2: #5C636D;
  --accent: #E8B26B;
  --accent-dim: #C8934E;
  --accent-cool: #6FB3D2;
  --success: #7FB893;
  --danger: #D97366;
  --grain: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2'/%3E%3CfeColorMatrix values='0 0 0 0 0.9 0 0 0 0 0.85 0 0 0 0 0.75 0 0 0 0.06 0'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
}

/* -------- base chrome -------- */
html, body, [class*="css"], .stApp {
  font-family: 'Inter', -apple-system, system-ui, sans-serif;
  font-feature-settings: 'ss01', 'cv11';
  color: var(--text-0);
  background: var(--bg-0);
}
.stApp {
  background:
    radial-gradient(ellipse 1200px 600px at 20% -10%, rgba(232, 178, 107, 0.08), transparent 60%),
    radial-gradient(ellipse 800px 500px at 85% 10%, rgba(111, 179, 210, 0.05), transparent 60%),
    var(--bg-0);
}
.stApp::before {
  content: '';
  position: fixed;
  inset: 0;
  background-image: var(--grain);
  opacity: 0.35;
  pointer-events: none;
  z-index: 0;
  mix-blend-mode: overlay;
}
#MainMenu, footer, header[data-testid="stHeader"] { display: none !important; }

.block-container {
  max-width: 1180px;
  padding: 3.5rem 2.5rem 6rem;
  position: relative;
  z-index: 1;
}

/* -------- typography -------- */
h1, h2, h3 { color: var(--text-0); letter-spacing: -0.02em; font-weight: 500; }
.stMarkdown p, .stMarkdown li { color: var(--text-1); line-height: 1.6; }

/* -------- hero -------- */
.hero {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  padding: 1rem 0 2.5rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 3rem;
}
.hero-left .eyebrow {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--accent);
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
  gap: 0.6rem;
}
.hero-left .eyebrow::before {
  content: '';
  width: 24px;
  height: 1px;
  background: var(--accent);
}
.hero-title {
  font-family: 'Fraunces', serif;
  font-weight: 300;
  font-size: clamp(2.8rem, 5.5vw, 4.5rem);
  line-height: 0.95;
  letter-spacing: -0.03em;
  color: var(--text-0);
  margin: 0;
}
.hero-title em {
  font-style: italic;
  color: var(--accent);
  font-weight: 300;
}
.hero-sub {
  font-family: 'Inter', sans-serif;
  font-size: 14px;
  color: var(--text-1);
  max-width: 420px;
  margin-top: 1.25rem;
  line-height: 1.55;
}
.hero-right {
  text-align: right;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: var(--text-2);
  letter-spacing: 0.08em;
}
.hero-right .version {
  color: var(--text-1);
  font-size: 12px;
  margin-bottom: 0.4rem;
}
.status-pill {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.4rem 0.9rem;
  margin-top: 0.6rem;
  border: 1px solid var(--border-bright);
  border-radius: 999px;
  background: var(--bg-1);
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--text-1);
}
.status-pill .dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--success);
  box-shadow: 0 0 8px var(--success);
}

/* -------- numbered section labels -------- */
.section-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--text-2);
  margin: 2.5rem 0 1rem;
  display: flex;
  align-items: center;
  gap: 0.75rem;
}
.section-label .num {
  color: var(--accent);
  font-weight: 500;
}
.section-label .line {
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, var(--border), transparent);
}

/* -------- cards -------- */
.card {
  background: linear-gradient(180deg, var(--bg-1) 0%, rgba(19, 22, 27, 0.6) 100%);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.75rem;
  margin-bottom: 1rem;
  box-shadow:
    0 1px 0 rgba(255, 255, 255, 0.02) inset,
    0 20px 60px -30px rgba(0, 0, 0, 0.6);
}

/* -------- Streamlit widget overrides -------- */
/* Radio buttons as filmstrip */
div[data-testid="stRadio"] > div {
  display: flex;
  gap: 0;
  background: var(--bg-1);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 4px;
}
div[data-testid="stRadio"] label {
  flex: 1;
  display: flex;
  justify-content: center;
  padding: 0.75rem 1rem;
  border-radius: 7px;
  cursor: pointer;
  transition: all 0.2s;
  color: var(--text-1) !important;
  font-size: 13px !important;
  letter-spacing: 0.02em;
}
div[data-testid="stRadio"] label:has(input:checked) {
  background: var(--bg-3);
  color: var(--text-0) !important;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4), 0 0 0 1px var(--border-bright);
}
div[data-testid="stRadio"] label > div:first-child { display: none; }

/* Text inputs — editorial bottom-border style */
.stTextInput input, .stTextArea textarea, .stNumberInput input {
  background: transparent !important;
  border: none !important;
  border-bottom: 1px solid var(--border-bright) !important;
  border-radius: 0 !important;
  color: var(--text-0) !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 14px !important;
  padding: 0.6rem 0.2rem !important;
  transition: border-color 0.2s;
}
.stTextInput input:focus, .stTextArea textarea:focus, .stNumberInput input:focus {
  border-bottom-color: var(--accent) !important;
  box-shadow: none !important;
  outline: none !important;
}
.stTextInput label, .stTextArea label, .stNumberInput label, .stSelectbox label, .stFileUploader label {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 10px !important;
  text-transform: uppercase !important;
  letter-spacing: 0.18em !important;
  color: var(--text-2) !important;
  font-weight: 500 !important;
  margin-bottom: 0.35rem !important;
}

/* Selectbox */
.stSelectbox div[data-baseweb="select"] > div {
  background: transparent !important;
  border: none !important;
  border-bottom: 1px solid var(--border-bright) !important;
  border-radius: 0 !important;
  color: var(--text-0) !important;
}

/* File uploader */
.stFileUploader section {
  background: var(--bg-1) !important;
  border: 1px dashed var(--border-bright) !important;
  border-radius: 10px !important;
  padding: 2rem !important;
  transition: border-color 0.2s, background 0.2s;
}
.stFileUploader section:hover {
  border-color: var(--accent) !important;
  background: var(--bg-2) !important;
}
.stFileUploader small { color: var(--text-2) !important; }

/* Checkbox */
.stCheckbox label {
  color: var(--text-1) !important;
  font-size: 13px !important;
}
.stCheckbox label:hover { color: var(--text-0) !important; }

/* Primary button — amber pill, lift on hover */
.stButton > button {
  background: var(--accent) !important;
  color: #0B0D10 !important;
  border: none !important;
  border-radius: 999px !important;
  padding: 0.95rem 2rem !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important;
  font-size: 13px !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
  box-shadow: 0 4px 20px rgba(232, 178, 107, 0.15) !important;
}
.stButton > button:hover {
  background: #F2BD78 !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 8px 30px rgba(232, 178, 107, 0.35) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* Tabs — filmstrip */
div[data-testid="stTabs"] button[data-baseweb="tab"] {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 11px !important;
  text-transform: uppercase !important;
  letter-spacing: 0.15em !important;
  color: var(--text-2) !important;
  background: transparent !important;
  border-bottom: 1px solid var(--border) !important;
  padding: 1rem 1.5rem !important;
}
div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
  color: var(--accent) !important;
  border-bottom: 1px solid var(--accent) !important;
}
div[data-testid="stTabs"] div[data-baseweb="tab-highlight"] { display: none !important; }
div[data-testid="stTabs"] div[data-baseweb="tab-border"] { background: var(--border) !important; }

/* Expanders */
div[data-testid="stExpander"] {
  background: var(--bg-1) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  margin-bottom: 0.75rem !important;
}
div[data-testid="stExpander"] summary {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 11px !important;
  letter-spacing: 0.15em !important;
  text-transform: uppercase !important;
  color: var(--text-1) !important;
  padding: 1rem 1.25rem !important;
}

/* Dividers */
hr { border-color: var(--border) !important; margin: 2.5rem 0 !important; }

/* Code blocks — terminal style */
.stCode, pre {
  background: #07090C !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 12px !important;
  color: #A8B0BA !important;
  padding: 1.25rem !important;
  position: relative;
}
.stCode::before, pre::before {
  content: '';
  display: block;
  position: absolute;
  top: 0.7rem;
  left: 1rem;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: var(--danger);
  box-shadow:
    18px 0 0 #D9A066,
    36px 0 0 var(--success);
}
.stCode { padding-top: 2.5rem !important; }

/* Alerts */
div[data-testid="stAlert"] {
  background: var(--bg-1) !important;
  border: 1px solid var(--border) !important;
  border-left: 3px solid var(--accent) !important;
  border-radius: 6px !important;
  color: var(--text-0) !important;
}

/* Captions */
.stCaption, [data-testid="stCaptionContainer"] {
  color: var(--text-2) !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 11px !important;
  letter-spacing: 0.05em !important;
}

/* Images */
[data-testid="stImage"] img {
  border-radius: 8px;
  border: 1px solid var(--border);
  box-shadow: 0 20px 60px -30px rgba(0, 0, 0, 0.8);
}

/* Subheaders */
.stApp h2, .stApp h3 {
  font-family: 'Fraunces', serif;
  font-weight: 400;
  letter-spacing: -0.01em;
}

/* Markdown headers used as labels */
.muted { color: var(--text-2); font-size: 12px; font-family: 'JetBrains Mono', monospace; letter-spacing: 0.1em; }

/* Footer */
.app-footer {
  margin-top: 5rem;
  padding-top: 2rem;
  border-top: 1px solid var(--border);
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: var(--text-2);
}
.app-footer em {
  font-family: 'Fraunces', serif;
  font-style: italic;
  font-size: 12px;
  color: var(--text-1);
  text-transform: none;
  letter-spacing: 0;
}
</style>
"""

st.markdown(CINEMATIC_CSS, unsafe_allow_html=True)

# =============================================================================
# HERO
# =============================================================================

st.markdown(
    """
    <div class="hero">
      <div class="hero-left">
        <div class="eyebrow">MEDICAL LONGFORM · PIPELINE v2</div>
        <h1 class="hero-title">Cinematic explainers,<br><em>rendered from a PDF.</em></h1>
        <div class="hero-sub">
          PDFs and topics become production-ready videos: narration-faithful segmenting,
          character-locked imagery via Nano Banana Pro, Kling motion, ElevenLabs voice, Remotion compose.
        </div>
      </div>
      <div class="hero-right">
        <div class="version">BUILD · NANO_BANANA_PRO</div>
        <div>16:9 · 2K · 150 WPM · 5s CLIPS</div>
        <div class="status-pill"><span class="dot"></span>READY</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# 01 — SOURCE
# =============================================================================

st.markdown(
    '<div class="section-label"><span class="num">01</span> SOURCE<div class="line"></div></div>',
    unsafe_allow_html=True,
)

input_mode = st.radio(
    "input_mode",
    ["Topic", "PDF File", "Creative Brief"],
    horizontal=True,
    label_visibility="collapsed",
)

topic = ""
pdf_file = None
brief_text = ""

if input_mode == "Topic":
    topic = st.text_input(
        "Topic",
        placeholder="e.g. Left heart failure & secondary pulmonary hypertension",
    )
elif input_mode == "PDF File":
    pdf_file = st.file_uploader("Upload medical education PDF", type=["pdf"])
else:
    brief_text = st.text_area(
        "Creative brief",
        placeholder=(
            "A 3-minute prestige explainer on how beta blockers work.\n\n"
            "Voice: calm, measured, documentary narrator.\n"
            "Aesthetic: warm monitor glow, teal-amber grade.\n"
            "Character: 62yo woman, stoic, hospital gown.\n"
        ),
        height=180,
    )

# =============================================================================
# 02 — DIRECTION
# =============================================================================

st.markdown(
    '<div class="section-label"><span class="num">02</span> DIRECTION<div class="line"></div></div>',
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2, gap="large")

with col1:
    duration = st.number_input(
        "Duration (minutes)",
        min_value=0.1,
        max_value=30.0,
        value=3.0,
        step=0.5,
        format="%.1f",
    )
    pipeline_mode = st.selectbox(
        "Pipeline mode",
        ["auto", "creative", "production"],
        index=0,
        help="auto → production for PDFs, creative for topic-only",
    )
    voice_id = st.text_input(
        "ElevenLabs voice ID",
        value=os.environ.get("ELEVENLABS_VOICE_ID", ""),
    )

with col2:
    avatar_path = str(DEFAULT_AVATAR) if DEFAULT_AVATAR.exists() else ""
    avatar_image = st.text_input("Avatar reference (face)", value=avatar_path)
    character_image = st.text_input(
        "Character override (optional)",
        placeholder="Path to a patient headshot — overrides auto-generated sheet",
    )
    style_image = st.text_input(
        "Style reference (optional)",
        placeholder="Path to a still that defines the look",
    )

if duration < 1.0:
    st.warning("Under 1 min may produce short, low-density scripts.")
elif duration > 15.0:
    st.warning("Over 15 min will take a long time to render all assets.")

# =============================================================================
# 03 — RENDER PIPELINE
# =============================================================================

st.markdown(
    '<div class="section-label"><span class="num">03</span> RENDER PIPELINE<div class="line"></div></div>',
    unsafe_allow_html=True,
)

skip_cols = st.columns(6)
skip_images = skip_cols[0].checkbox("Skip images")
skip_voice = skip_cols[1].checkbox("Skip voice")
skip_avatar = skip_cols[2].checkbox("Skip avatar")
skip_animation = skip_cols[3].checkbox("Skip animation")
skip_compose = skip_cols[4].checkbox("Skip compose")
dry_run = skip_cols[5].checkbox("Dry run", value=True, help="Skip Airtable push")

st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

# =============================================================================
# RUN BUTTON
# =============================================================================

run_clicked = st.button("◐  INITIATE RENDER", type="primary", use_container_width=True)

if run_clicked:
    cmd = [sys.executable, "run.py"]

    if input_mode == "Topic":
        if not topic:
            st.error("Enter a topic before initiating render.")
            st.stop()
        cmd.extend(["--topic", topic])
    elif input_mode == "PDF File":
        if not pdf_file:
            st.error("Upload a PDF before initiating render.")
            st.stop()
        temp_pdf = PROJECT_ROOT / f"_uploaded_{pdf_file.name}"
        temp_pdf.write_bytes(pdf_file.read())
        cmd.append(str(temp_pdf))
    else:
        if not brief_text:
            st.error("Enter a creative brief before initiating render.")
            st.stop()
        cmd.extend(["--brief", brief_text])

    cmd.extend(["--duration", str(duration)])
    if voice_id:
        cmd.extend(["--voice-id", voice_id])
    if avatar_image:
        cmd.extend(["--avatar-image", avatar_image])
    if character_image:
        cmd.extend(["--character-image", character_image])
    if style_image:
        cmd.extend(["--style-image", style_image])
    if skip_images:
        cmd.append("--skip-images")
    if skip_voice:
        cmd.append("--skip-voice")
    if skip_avatar:
        cmd.append("--skip-avatar")
    if skip_animation:
        cmd.append("--skip-animation")
    if skip_compose:
        cmd.append("--skip-compose")
    if dry_run:
        cmd.append("--dry-run")
    if pipeline_mode != "auto":
        cmd.extend(["--mode", pipeline_mode])

    st.markdown(
        '<div class="section-label"><span class="num">04</span> LOG STREAM<div class="line"></div></div>',
        unsafe_allow_html=True,
    )
    st.code(f"$ {' '.join(cmd)}", language="bash")

    log_container = st.empty()
    output_lines = []

    with st.spinner("Rendering…"):
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(PROJECT_ROOT),
            bufsize=1,
        )
        for line in process.stdout:
            output_lines.append(line)
            log_container.code("".join(output_lines), language="text")
        process.wait()

    if process.returncode == 0:
        st.success("Render complete.")
    else:
        st.error(f"Render failed · exit code {process.returncode}")

    if input_mode == "PDF File" and pdf_file:
        temp_pdf = PROJECT_ROOT / f"_uploaded_{pdf_file.name}"
        if temp_pdf.exists():
            temp_pdf.unlink()

# =============================================================================
# ARCHIVE — generated assets browser
# =============================================================================

st.markdown(
    '<div class="section-label"><span class="num">05</span> ARCHIVE<div class="line"></div></div>',
    unsafe_allow_html=True,
)

if OUTPUT_DIR.exists():
    projects = sorted(
        [d for d in OUTPUT_DIR.iterdir() if d.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not projects:
        st.info("No renders yet. Initiate one above to see it archived here.")
    else:
        proj_col, run_col = st.columns([2, 1])
        with proj_col:
            selected = st.selectbox("Project", [p.name for p in projects])
        project_dir = OUTPUT_DIR / selected

        run_dirs = list_project_runs(project_dir)
        if run_dirs:
            latest_run_id = get_latest_run_id(project_dir)
            run_names = [run_dir.name for run_dir in run_dirs]
            default_index = run_names.index(latest_run_id) if latest_run_id in run_names else 0
            with run_col:
                selected_run = st.selectbox("Run", run_names, index=default_index)
            artifact_dir = project_dir / "runs" / selected_run
        else:
            artifact_dir = project_dir
            with run_col:
                st.markdown("<div class='muted' style='padding-top: 1.5rem'>LEGACY LAYOUT</div>", unsafe_allow_html=True)

        st.markdown(
            f"<div class='muted' style='margin: 0.5rem 0 1.5rem'>PATH · {artifact_dir}</div>",
            unsafe_allow_html=True,
        )

        # Script
        script_path = artifact_dir / "script.json"
        if script_path.exists():
            with st.expander("SCRIPT", expanded=False):
                script_data = json.loads(script_path.read_text())
                meta_cols = st.columns(4)
                meta_cols[0].metric("Topic", (script_data.get("topic", "") or "—")[:24])
                meta_cols[1].metric("Duration", f"{script_data.get('total_minutes', 0)} min")
                meta_cols[2].metric("Words", script_data.get("total_word_count", 0))
                meta_cols[3].metric("Mode", script_data.get("pipeline_mode", "") or "legacy")
                st.markdown("---")
                for scene in script_data.get("scenes", []):
                    st.markdown(
                        f"<div class='muted'>SCENE · {scene['scene']} · {scene['word_count']} WORDS</div>",
                        unsafe_allow_html=True,
                    )
                    st.text(scene["script_full"][:500])
                    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

        review_path = artifact_dir / "review" / "script_review.json"
        if review_path.exists():
            with st.expander("REVIEW", expanded=False):
                review_data = json.loads(review_path.read_text())
                st.write(f"**Approved:** {review_data.get('approved')}")
                st.write(f"**Grounded:** {review_data.get('grounded')}")
                st.write(f"**Summary:** {review_data.get('summary', '')}")
                for blocker in review_data.get("blockers", []):
                    st.markdown(f"- {blocker}")
                for warning in review_data.get("warnings", []):
                    st.markdown(f"- {warning}")

        tabs = st.tabs(["FINAL CUT", "FRAMES", "VOICE", "AVATARS", "MOTION", "CHARACTER"])

        with tabs[0]:
            final_video = artifact_dir / "final_video.mp4"
            if final_video.exists():
                st.video(str(final_video))
                st.caption(f"{final_video.stat().st_size / (1024 * 1024):.1f} MB · {final_video.name}")
            else:
                st.info("No final cut yet. Run the full pipeline to compose.")

        with tabs[1]:
            images_dir = artifact_dir / "images"
            if images_dir.exists():
                imgs = sorted(images_dir.glob("*.png"))
                if imgs:
                    grid = st.columns(3)
                    for i, img in enumerate(imgs):
                        with grid[i % 3]:
                            st.image(str(img), caption=img.name, use_container_width=True)
                else:
                    st.info("No frames rendered yet.")
            else:
                st.info("No frames rendered yet.")

        with tabs[2]:
            voice_dir = artifact_dir / "voice"
            if voice_dir.exists():
                for mp3 in sorted(voice_dir.glob("*.mp3")):
                    st.audio(str(mp3), format="audio/mp3")
                    st.caption(mp3.name)
            else:
                st.info("No voice yet.")

        with tabs[3]:
            avatar_dir = artifact_dir / "avatars"
            if avatar_dir.exists():
                for vid in sorted(avatar_dir.glob("*.mp4")):
                    st.video(str(vid))
                    st.caption(vid.name)
            else:
                st.info("No avatars yet.")

        with tabs[4]:
            anim_dir = artifact_dir / "animations"
            if anim_dir.exists():
                grid = st.columns(2)
                anims = sorted(anim_dir.glob("*.mp4"))
                for i, vid in enumerate(anims):
                    with grid[i % 2]:
                        st.video(str(vid))
                        st.caption(vid.name)
                if not anims:
                    st.info("No motion clips yet.")
            else:
                st.info("No motion clips yet.")

        with tabs[5]:
            run_character = artifact_dir / "character" / "character.png"
            latest_character = project_dir / "character" / "latest" / "character.png"
            char_spec_path = artifact_dir / "character" / "character.json"

            char_cols = st.columns([1, 1])
            with char_cols[0]:
                if run_character.exists():
                    st.image(str(run_character), caption="Run character sheet", use_container_width=True)
                elif latest_character.exists():
                    st.image(str(latest_character), caption="Latest project character sheet", use_container_width=True)
                else:
                    st.info("No character sheet yet.")
            with char_cols[1]:
                if char_spec_path.exists():
                    spec = json.loads(char_spec_path.read_text())
                    st.markdown("<div class='muted'>CANONICAL SPEC</div>", unsafe_allow_html=True)
                    for key in ("one_line", "age", "sex", "ethnicity", "skin_tone", "build", "hair", "wardrobe", "demeanor"):
                        val = spec.get(key, "")
                        if val:
                            st.markdown(f"**{key.replace('_', ' ').title()}** · {val}")
else:
    st.info("No archive yet. The output/ directory will appear after the first render.")

# =============================================================================
# FOOTER
# =============================================================================

st.markdown(
    """
    <div class="app-footer">
      <div>CLAUDE · NANO BANANA PRO · KLING · ELEVENLABS · REMOTION</div>
      <div><em>made for medical storytelling</em></div>
    </div>
    """,
    unsafe_allow_html=True,
)
