import json
from pathlib import Path

from tools.project_store import (
    create_run_context,
    export_latest_character_assets,
    prune_project_runs,
    resolve_existing_script_path,
    set_latest_run,
    write_project_manifest,
)


def test_run_context_and_latest_resolution(tmp_path: Path):
    ctx = create_run_context('Left Heart Failure', output_root=tmp_path, run_id='20260407_090000')
    script_path = ctx.run_dir / 'script.json'
    script_path.write_text(json.dumps({'project_name': ctx.project_name, 'run_id': ctx.run_id}))

    write_project_manifest(ctx.project_dir, ctx.project_name, ctx.run_id)
    set_latest_run(ctx.project_dir, ctx.run_id)

    resolved = resolve_existing_script_path(ctx.project_name, output_root=tmp_path)
    assert resolved == script_path


def test_character_export_creates_project_level_latest(tmp_path: Path):
    ctx = create_run_context('Pulmonary Hypertension', output_root=tmp_path, run_id='20260407_091500')
    (ctx.character_dir / 'character.png').write_bytes(b'png')
    (ctx.character_dir / 'character.json').write_text('{}')

    export_dir = export_latest_character_assets(ctx)

    assert export_dir is not None
    assert (export_dir / 'character.png').exists()
    assert (export_dir / 'character.json').exists()


def test_prune_project_runs_keeps_newest_two(tmp_path: Path):
    run_ids = [
        '20260407_090000',
        '20260407_091000',
        '20260407_092000',
        '20260407_093000',
    ]
    last_ctx = None
    for run_id in run_ids:
        ctx = create_run_context('Pulmonary Hypertension', output_root=tmp_path, run_id=run_id)
        (ctx.run_dir / 'script.json').write_text(json.dumps({'run_id': run_id}))
        write_project_manifest(ctx.project_dir, ctx.project_name, ctx.run_id)
        set_latest_run(ctx.project_dir, ctx.run_id)
        last_ctx = ctx

    assert last_ctx is not None
    deleted = prune_project_runs(last_ctx.project_dir, keep=2)

    remaining = [p.name for p in sorted((last_ctx.project_dir / 'runs').iterdir()) if p.is_dir()]
    assert remaining == ['20260407_092000', '20260407_093000']
    assert [p.name for p in deleted] == ['20260407_091000', '20260407_090000']

    latest = json.loads((last_ctx.project_dir / 'latest_run.json').read_text())
    assert latest['run_id'] == '20260407_093000'
