"""Airtable setup validator and configurator.

Checks your Airtable base for the required tables and fields,
reports what's missing, and updates the codebase with correct table IDs.

Usage:
    python setup_airtable.py              # Check tables and fields
    python setup_airtable.py --fix        # Auto-update table IDs in airtable_client.py
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()

# What the pipeline needs
REQUIRED_TABLES = {
    "Projects": {
        "fields": {
            "Project Name": "singleLineText",
            "Status": "singleSelect",
            "INPUT_Request": "multilineText",
            "INPUT_voice_id": "singleLineText",
            "INPUT_Image_1": "multipleAttachments",
            "INPUT_Image_2": "multipleAttachments",
            "Source_PDF": "multipleAttachments",
            "Total_Minutes": "number",
            "aspect_ratio": "singleLineText",
        },
        "select_options": {
            "Status": ["Create", "Processing", "Script_Done", "Done", "Error"],
        },
    },
    "Scenes": {
        "fields": {
            "Project Name": "singleLineText",
            "scene": "singleLineText",
            "estimate_mins": "number",
            "script": "multilineText",
            "script_full": "multilineText",
            "speech_prompt": "multilineText",
            "visual_summary": "multilineText",
            "Status_voice": "singleSelect",
            "scene_voice": "multipleAttachments",
            "Status_broll": "singleSelect",
            "scene_video": "multipleAttachments",
            "Status_animation": "singleSelect",
            "scene_animation": "multipleAttachments",
        },
        "select_options": {
            "Status_voice": ["Create", "Done", "Skip"],
            "Status_broll": ["Create", "Done", "Skip"],
            "Status_animation": ["Create", "Done", "Skip"],
        },
    },
    "Segments": {
        "fields": {
            "Project Name": "singleLineText",
            "Scene Name": "singleLineText",
            "segment": "singleLineText",
            "image_prompt": "multilineText",
            "video_prompt": "multilineText",
            "Status_image": "singleSelect",
            "segment_image": "multipleAttachments",
            "Status_video": "singleSelect",
            "segment_video": "multipleAttachments",
        },
        "select_options": {
            "Status_image": ["Create", "Done", "Skip"],
            "Status_video": ["Create", "Done", "Skip"],
        },
    },
}

# Airtable metadata API
META_BASE = "https://api.airtable.com/v0/meta/bases"


def get_base_schema(pat: str, base_id: str) -> dict:
    """Fetch the full schema for a base."""
    response = httpx.get(
        f"{META_BASE}/{base_id}/tables",
        headers={"Authorization": f"Bearer {pat}"},
        timeout=15.0,
    )
    if response.status_code == 401:
        print("❌ Authentication failed. Check your AIRTABLE_PAT.")
        print("   Get one at: https://airtable.com/create/tokens")
        print("   Scopes needed: data.records:read, data.records:write, schema.bases:read")
        sys.exit(1)
    if response.status_code == 403:
        print("❌ Access denied. Your token needs the 'schema.bases:read' scope.")
        print("   Edit your token at: https://airtable.com/create/tokens")
        sys.exit(1)
    response.raise_for_status()
    return response.json()


def check_tables(schema: dict) -> dict:
    """Compare base schema against requirements. Returns table ID mapping."""
    tables_in_base = {t["name"]: t for t in schema.get("tables", [])}
    table_ids = {}
    all_good = True

    for req_name, req_spec in REQUIRED_TABLES.items():
        print(f"\n📋 Table: {req_name}")

        # Try exact match, case-insensitive, then singular/plural variants
        table = tables_in_base.get(req_name)
        if not table:
            for name, t in tables_in_base.items():
                if (name.lower() == req_name.lower()
                    or name.lower().rstrip("s") == req_name.lower().rstrip("s")):
                    table = t
                    break

        if not table:
            print(f"   ❌ NOT FOUND — create this table in your Airtable base")
            print(f"   Required fields:")
            for fname, ftype in req_spec["fields"].items():
                print(f"      - {fname} ({ftype})")
            all_good = False
            continue

        table_ids[req_name] = table["id"]
        print(f"   ✓ Found (ID: {table['id']})")

        # Check fields
        existing_fields = {f["name"]: f for f in table.get("fields", [])}
        missing = []
        found = []

        for fname, ftype in req_spec["fields"].items():
            if fname in existing_fields:
                actual_type = existing_fields[fname].get("type", "?")
                if actual_type == ftype:
                    found.append(fname)
                else:
                    # Close enough — some types are compatible
                    found.append(fname)
                    print(f"   ⚠️  {fname}: expected {ftype}, got {actual_type} (may still work)")
            else:
                missing.append((fname, ftype))

        if found:
            print(f"   ✓ {len(found)}/{len(req_spec['fields'])} fields present")

        if missing:
            all_good = False
            print(f"   ❌ Missing {len(missing)} field(s):")
            for fname, ftype in missing:
                print(f"      - {fname} ({ftype})")

    if all_good:
        print(f"\n✅ All tables and fields look good!")
    else:
        print(f"\n⚠️  Some tables or fields are missing — create them in Airtable")

    return table_ids


def update_client_ids(table_ids: dict) -> None:
    """Update the hardcoded table IDs in airtable_client.py."""
    client_path = os.path.join(os.path.dirname(__file__), "tools", "airtable_client.py")

    with open(client_path, "r") as f:
        content = f.read()

    original = content

    if "Projects" in table_ids:
        content = re.sub(
            r'PROJECT_TABLE = "[^"]*"',
            f'PROJECT_TABLE = "{table_ids["Projects"]}"',
            content,
        )
    if "Scenes" in table_ids:
        content = re.sub(
            r'SCENES_TABLE = "[^"]*"',
            f'SCENES_TABLE = "{table_ids["Scenes"]}"',
            content,
        )
    if "Segments" in table_ids:
        content = re.sub(
            r'SEGMENTS_TABLE = "[^"]*"',
            f'SEGMENTS_TABLE = "{table_ids["Segments"]}"',
            content,
        )

    if content != original:
        with open(client_path, "w") as f:
            f.write(content)
        print(f"\n🔧 Updated table IDs in tools/airtable_client.py:")
        for name, tid in table_ids.items():
            print(f"   {name} → {tid}")
    else:
        print(f"\n✓ Table IDs already up to date")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and configure Airtable base")
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Auto-update table IDs in airtable_client.py",
    )
    parser.add_argument(
        "--base-id",
        type=str,
        default="",
        help="Override AIRTABLE_BASE_ID from .env",
    )
    args = parser.parse_args()

    pat = os.environ.get("AIRTABLE_PAT", "")
    base_id = args.base_id or os.environ.get("AIRTABLE_BASE_ID", "")

    if not pat:
        print("❌ AIRTABLE_PAT not set in .env")
        print("\nTo get one:")
        print("  1. Go to https://airtable.com/create/tokens")
        print("  2. Create a token with scopes: data.records:read, data.records:write, schema.bases:read")
        print("  3. Give it access to your base")
        print("  4. Add to .env: AIRTABLE_PAT=pat.xxxxx...")
        sys.exit(1)

    if not base_id:
        print("❌ AIRTABLE_BASE_ID not set in .env")
        print("   Find it in your Airtable base URL: airtable.com/appXXXXXXX/...")
        sys.exit(1)

    print(f"🔍 Checking Airtable base: {base_id}")
    schema = get_base_schema(pat, base_id)

    print(f"   Found {len(schema.get('tables', []))} tables:")
    for t in schema.get("tables", []):
        print(f"   - {t['name']} ({t['id']}, {len(t.get('fields', []))} fields)")

    table_ids = check_tables(schema)

    if args.fix and table_ids:
        update_client_ids(table_ids)

    # Show .env status
    print(f"\n--- .env Status ---")
    env_keys = [
        "AIRTABLE_PAT", "AIRTABLE_BASE_ID", "OPENAI_API_KEY",
        "ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID",
        "GEMINI_API_KEY", "WAVESPEED_API_KEY", "KIE_API_KEY",
    ]
    for key in env_keys:
        val = os.environ.get(key, "")
        status = "✓ set" if val else "❌ missing"
        print(f"   {key}: {status}")


if __name__ == "__main__":
    main()
