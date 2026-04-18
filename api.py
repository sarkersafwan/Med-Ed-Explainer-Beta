import asyncio
import json
import logging
import subprocess
import sys
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Med-Ed Explainer API", description="Local backend for Board Buddy Extension")

# Allow Chrome Extension to talk to the local API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionPayload(BaseModel):
    questionText: str
    options: list[str] = []
    correctAnswer: str = ""
    explanation: str = ""

def run_pipeline_script(brief_text: str):
    """Executes the run.py pipeline as a detached subprocess."""
    command = [
        sys.executable,
        "run.py",
        "--brief",
        brief_text,
        "--duration",
        "1", # Use 1 min by default for Board Buddy extensions
        "--avatar-image",
        "MedVidSpeaker.png"
    ]
    
    logger.info(f"Starting pipeline with command: {' '.join(command)}")
    
    # Run in subprocess and capture output to a log file
    try:
        log_file = open("api_worker.log", "a")
        log_file.write(f"\n\n--- Starting new run ---\n")
        log_file.flush()
        subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
        logger.info("Pipeline triggered successfully in the background.")
    except Exception as e:
        logger.error(f"Failed to start pipeline: {e}")

@app.post("/generate-video")
async def generate_video(payload: QuestionPayload, background_tasks: BackgroundTasks):
    """Receives extracted question from Board Buddy and spawns the Video pipeline."""
    logger.info("Received request to generate video from Board Buddy.")
    
    # Format the entire payload into a comprehensive creative brief
    brief_parts = [
        "Create a highly engaging medical education explainer video covering the following clinical scenario:",
        f"\nCLINICAL VIGNETTE:\n{payload.questionText}"
    ]
    
    if payload.options:
        brief_parts.append("\nANSWER CHOICES:")
        for opt in payload.options:
            brief_parts.append(f"- {opt}")
            
    if payload.correctAnswer:
        brief_parts.append(f"\nCORRECT ANSWER: {payload.correctAnswer}")
        
    if payload.explanation:
        brief_parts.append(f"\nEXPLANATION & PATHOPHYSIOLOGY:\n{payload.explanation}")
        
    brief_text = "\n".join(brief_parts)
    
    # Schedule the background script to run immediately after we respond to Chrome
    background_tasks.add_task(run_pipeline_script, brief_text)
    
    return {"status": "started", "message": "Video generation pipeline has been triggered."}

if __name__ == "__main__":
    import uvicorn
    # Make sure to run this via `python api.py` or `uvicorn api:app --port 5030`
    uvicorn.run("api:app", host="127.0.0.1", port=5030, log_level="info")
