"""Helpers for run-scoped output storage and project metadata."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from shutil import copy2, rmtree

LATEST_RUN_FILE = "latest_run.json"
PROJECT_MANIFEST_FILE = "project.json"
DEFAULT_MAX_PROJECT_RUNS = 2


@dataclass(frozen=True)
class RunContext:
    """Filesystem paths for a single pipeline run."""

    project_name: str
    project_dir: Path
    runs_dir: Path
    run_id: str
    run_dir: Path

    @property
    def character_dir(self) -> Path:
        return self.run_dir / "character"

    @property
    def review_dir(self) -> Path:
        return self.run_dir / "review"

    @property
    def evidence_dir(self) -> Path:
        return self.run_dir / "evidence"


def slugify_project_name(value: str) -> str:
    """Generate a filesystem-safe project slug."""
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "untitled_project"


def generate_run_id(now: datetime | None = None) -> str:
    """Generate a sortable run identifier."""
    stamp = now or datetime.now().astimezone()
    return stamp.strftime("%Y%m%d_%H%M%S")


def create_run_context(
    project_name: str,
    output_root: Path = Path("output"),
    run_id: str | None = None,
) -> RunContext:
    """Create directories for a new run and return the resolved paths."""
    slug = slugify_project_name(project_name)
    project_dir = output_root / slug
    runs_dir = project_dir / "runs"
    resolved_run_id = run_id or generate_run_id()
    run_dir = runs_dir / resolved_run_id

    for path in (project_dir, runs_dir, run_dir, run_dir / "images", run_dir / "voice", run_dir / "avatars", run_dir / "animations", run_dir / "character", run_dir / "review", run_dir / "evidence"):
        path.mkdir(parents=True, exist_ok=True)

    return RunContext(
        project_name=slug,
        project_dir=project_dir,
        runs_dir=runs_dir,
        run_id=resolved_run_id,
        run_dir=run_dir,
    )


def write_project_manifest(project_dir: Path, project_name: str, run_id: str) -> None:
    """Persist lightweight project metadata for UI discovery."""
    manifest = {
        "project_name": project_name,
        "latest_run_id": run_id,
        "updated_at": datetime.now().astimezone().isoformat(),
    }
    (project_dir / PROJECT_MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))


def set_latest_run(project_dir: Path, run_id: str) -> None:
    """Point a project at its latest completed run."""
    payload = {
        "run_id": run_id,
        "updated_at": datetime.now().astimezone().isoformat(),
    }
    (project_dir / LATEST_RUN_FILE).write_text(json.dumps(payload, indent=2))


def get_latest_run_id(project_dir: Path) -> str | None:
    """Read the latest run identifier if present."""
    latest_file = project_dir / LATEST_RUN_FILE
    if latest_file.exists():
        data = json.loads(latest_file.read_text())
        run_id = data.get("run_id", "")
        if run_id:
            return run_id

    runs_dir = project_dir / "runs"
    if not runs_dir.exists():
        return None

    candidates = sorted(
        [p.name for p in runs_dir.iterdir() if p.is_dir()],
        reverse=True,
    )
    return candidates[0] if candidates else None


def get_run_dir(project_name: str, run_id: str, output_root: Path = Path("output")) -> Path:
    """Resolve a run directory from its project name and run id."""
    return output_root / slugify_project_name(project_name) / "runs" / run_id


def resolve_existing_script_path(
    project_name: str,
    run_id: str = "",
    output_root: Path = Path("output"),
) -> Path | None:
    """Find an existing script.json for a project, preferring a specific/latest run."""
    project_dir = output_root / slugify_project_name(project_name)
    if not project_dir.exists():
        return None

    if run_id:
        candidate = project_dir / "runs" / run_id / "script.json"
        if candidate.exists():
            return candidate

    latest_run_id = get_latest_run_id(project_dir)
    if latest_run_id:
        latest_candidate = project_dir / "runs" / latest_run_id / "script.json"
        if latest_candidate.exists():
            return latest_candidate

    legacy_script = project_dir / "script.json"
    if legacy_script.exists():
        return legacy_script

    for candidate in sorted(project_dir.glob("runs/*/script.json"), reverse=True):
        return candidate

    return None


def list_project_runs(project_dir: Path) -> list[Path]:
    """List run directories for a project, newest first."""
    runs_dir = project_dir / "runs"
    if not runs_dir.exists():
        return []
    return sorted([p for p in runs_dir.iterdir() if p.is_dir()], reverse=True)


def prune_project_runs(project_dir: Path, keep: int = DEFAULT_MAX_PROJECT_RUNS) -> list[Path]:
    """Delete older run directories so only the newest `keep` runs remain."""
    keep = max(1, int(keep))
    runs = list_project_runs(project_dir)
    to_delete = runs[keep:]

    for run_dir in to_delete:
        rmtree(run_dir, ignore_errors=False)

    remaining = list_project_runs(project_dir)
    latest_file = project_dir / LATEST_RUN_FILE
    manifest_file = project_dir / PROJECT_MANIFEST_FILE
    latest_run_id = remaining[0].name if remaining else ""

    if latest_run_id:
        payload = {
            "run_id": latest_run_id,
            "updated_at": datetime.now().astimezone().isoformat(),
        }
        latest_file.write_text(json.dumps(payload, indent=2))

        if manifest_file.exists():
            try:
                manifest = json.loads(manifest_file.read_text())
            except json.JSONDecodeError:
                manifest = {}
            manifest["latest_run_id"] = latest_run_id
            manifest["updated_at"] = datetime.now().astimezone().isoformat()
            manifest_file.write_text(json.dumps(manifest, indent=2))
    else:
        if latest_file.exists():
            latest_file.unlink()

    return to_delete


def export_latest_character_assets(run_context: RunContext) -> Path | None:
    """Copy the current run's character assets into the project-level character folder."""
    if not run_context.character_dir.exists():
        return None

    files = list(run_context.character_dir.glob("character.*"))
    if not files:
        return None

    export_dir = run_context.project_dir / "character" / "latest"
    export_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        copy2(src, export_dir / src.name)
    return export_dir
