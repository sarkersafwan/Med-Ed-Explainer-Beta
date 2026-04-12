"""Airtable client for the medical education video pipeline.

Handles CRUD operations for Project, Scenes, and Segments tables.
Supports uploading generated assets as attachments.
"""

from __future__ import annotations

import os

import httpx
from dotenv import load_dotenv
from pyairtable import Api

from tools.models import ProductionScript, Segment

AIRTABLE_META_BASE = "https://api.airtable.com/v0/meta"

load_dotenv()

# Table IDs — auto-updated by setup_airtable.py --fix
PROJECT_TABLE = "tblYiND5DkrZhIlLq"
SCENES_TABLE = "tblipThhapetdSJdm"
SEGMENTS_TABLE = "tblc8oQyblzO8JXjV"


class AirtableClient:
    """Wrapper for Airtable API operations."""

    def __init__(self) -> None:
        pat = os.environ.get("AIRTABLE_PAT", "")
        base_id = os.environ.get("AIRTABLE_BASE_ID", "appjmdOqi7hTArDN6")
        if not pat:
            raise ValueError("AIRTABLE_PAT not set in environment")

        self.api = Api(pat)
        self.base_id = base_id
        self.projects = self.api.table(base_id, PROJECT_TABLE)
        self.scenes = self.api.table(base_id, SCENES_TABLE)
        self.segments = self.api.table(base_id, SEGMENTS_TABLE)

    def list_tables(self) -> list[str]:
        return ["Projects", "Scenes", "Segments"]

    def create_project(
        self,
        script: ProductionScript,
        voice_id: str = "",
        aspect_ratio: str = "16:9",
        input_request: str = "",
        avatar_image_url: str = "",
        style_image_url: str = "",
        character_image_url: str = "",
    ) -> str:
        """Create a project record and return its ID."""
        fields = {
            "Project Name": script.project_name,
            "Status": "Done",
            "INPUT_Request": input_request or f"Topic: {script.topic}\nSource: {script.source_pdf}",
            "INPUT_voice_id": voice_id,
            "Total_Minutes": script.total_minutes,
            "aspect_ratio": aspect_ratio,
        }
        if avatar_image_url:
            fields["INPUT_Image_1"] = [{"url": avatar_image_url}]
        if style_image_url:
            fields["INPUT_Image_2"] = [{"url": style_image_url}]
        if character_image_url:
            fields["character_image"] = [{"url": character_image_url}]

        record = self.projects.create(fields)
        return record["id"]

    def ensure_project_field(
        self,
        field_name: str,
        field_type: str = "multipleAttachments",
    ) -> bool:
        """Ensure a field exists on the Projects table. Returns True if it exists
        (or was created), False if we couldn't create it (e.g. PAT missing
        schema.bases:write scope).

        Uses the Airtable Meta API. Idempotent — safe to call every run.
        """
        pat = os.environ.get("AIRTABLE_PAT", "")
        headers = {"Authorization": f"Bearer {pat}", "Content-Type": "application/json"}

        # 1. Check whether the field already exists via GET /meta/bases/{id}/tables
        try:
            resp = httpx.get(
                f"{AIRTABLE_META_BASE}/bases/{self.base_id}/tables",
                headers=headers,
                timeout=30.0,
            )
            if resp.status_code == 200:
                tables = resp.json().get("tables", [])
                for t in tables:
                    if t.get("id") == PROJECT_TABLE:
                        for f in t.get("fields", []):
                            if f.get("name") == field_name:
                                return True
                        break
        except Exception as e:
            print(f"      ⚠️  Could not read Airtable schema: {e}")
            return False

        # 2. Create the field
        try:
            resp = httpx.post(
                f"{AIRTABLE_META_BASE}/bases/{self.base_id}/tables/{PROJECT_TABLE}/fields",
                headers=headers,
                json={"name": field_name, "type": field_type},
                timeout=30.0,
            )
            if resp.status_code in (200, 201):
                print(f"      ✓ Created Airtable field '{field_name}' on Projects table")
                return True
            if resp.status_code == 422 and "DUPLICATE" in resp.text.upper():
                return True  # race: someone else created it
            if resp.status_code == 403:
                print(
                    f"      ⚠️  PAT lacks schema.bases:write scope — add the field "
                    f"'{field_name}' (type: {field_type}) to the Projects table manually, "
                    f"or regenerate your Airtable PAT with schema.bases:write."
                )
                return False
            print(f"      ⚠️  Could not create field '{field_name}': {resp.status_code} {resp.text[:200]}")
            return False
        except Exception as e:
            print(f"      ⚠️  Field creation failed: {e}")
            return False

    def attach_to_project(self, record_id: str, field: str, url: str) -> None:
        """Attach a file URL to a project record's attachment field."""
        self.projects.update(record_id, {field: [{"url": url}]})

    def push_scenes(
        self,
        script: ProductionScript,
        trigger_voice: bool = True,
        trigger_images: bool = True,
    ) -> list[str]:
        """Push all scenes to Airtable. Returns list of created record IDs."""
        record_ids = []
        for scene in script.scenes:
            fields = {
                "Project Name": script.project_name,
                "scene": scene.scene,
                "estimate_mins": scene.duration_minutes,
                "script": scene.script,
                "script_full": scene.script_full,
                "speech_prompt": scene.speech_prompt or script.speech_prompt,
                "visual_summary": scene.visual_summary,
                "Status_voice": "Create" if trigger_voice else "Skip",
                "Status_broll": "Create" if trigger_images else "Skip",
            }
            record = self.scenes.create(fields)
            record_ids.append(record["id"])
        return record_ids

    def push_segments(
        self,
        project_name: str,
        segments: list[Segment],
    ) -> list[str]:
        """Push all segments to Airtable. Returns list of created record IDs."""
        record_ids = []
        for seg in segments:
            fields = {
                "Project Name": project_name,
                "Scene Name": f"{seg.scene_number}",
                "segment": seg.segment_title,
                "image_prompt": seg.image_prompt,
                "video_prompt": seg.video_prompt,
                "Status_image": "Create",
                "Status_video": "Create",
            }
            record = self.segments.create(fields)
            record_ids.append(record["id"])
        return record_ids

    def attach_to_scene(self, record_id: str, field: str, url: str) -> None:
        """Attach a file URL to a scene record's attachment field."""
        self.scenes.update(record_id, {field: [{"url": url}]})

    def attach_to_segment(self, record_id: str, field: str, url: str) -> None:
        """Attach a file URL to a segment record's attachment field."""
        self.segments.update(record_id, {field: [{"url": url}]})

    def update_scene_status(
        self, record_id: str, field: str, status: str
    ) -> None:
        self.scenes.update(record_id, {field: status})

    def update_segment_status(
        self, record_id: str, field: str, status: str
    ) -> None:
        self.segments.update(record_id, {field: status})

    def get_scenes_by_status(
        self, status_field: str, status_value: str
    ) -> list[dict]:
        formula = f"{{{status_field}}}='{status_value}'"
        return self.scenes.all(formula=formula)
