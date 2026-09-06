"""Run the fixed v4 recipe in a new output directory, or print it with --dry-run."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from reproduction.prepare import MANIFEST_PATH, digest, prepare, project_path, verify_manifest


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, help='Must not exist; relative paths are based on the checkout root')
    parser.add_argument('--epochs', type=int, choices=(1, 15), default=15,
        help='15 reproduces the fixed recipe; 1 is a smoke test, never a full reproduction')
    parser.add_argument('--prepared-dir', help='Prepared directory inside this checkout (default: reproduction/prepared_v2)')
    parser.add_argument('--dry-run', action='store_true', help='Verify/relocate inputs and print the command without training or creating the training output')
    args = parser.parse_args(argv)
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output = output.resolve()
    record = output.with_name(output.name + '.invocation.json')
    if output.exists() or record.exists():
        parser.error('Refusing to reuse or overwrite an existing output directory or invocation record: ' + str(output))
    try:
        manifest, prepared = prepare(args.prepared_dir)
        training = manifest['training']
        settings = list(training['arguments'])
        settings[settings.index('--epochs') + 1] = str(args.epochs)
        command = [sys.executable, '-X', 'utf8', '-B', '-u', str(project_path(training['entrypoint'])),
            '--init-checkpoint', str(project_path(training['initial_checkpoint'])),
            '--data-source', str(project_path(prepared['data_source'])), '--output-dir', str(output),
            '--local-models', str(project_path(training['local_models'])), *settings]
        for entry in prepared['constraints']:
            command += ['--policy-constraints', str(project_path(entry['path']))]
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    mode = 'full_15_epoch_recipe' if args.epochs == 15 else 'smoke_only_not_full_reproduction'
    invocation = dict(format='must5_reproduction_run_v1', mode=mode, epochs=args.epochs,
        manifest_sha256=digest(MANIFEST_PATH), project_root=str(PROJECT_ROOT), command=command,
        output_dir=str(output), constraints=prepared['constraints'], expected=training['expected'],
        data_source=prepared['data_source'], cache_relocation=prepared['cache_relocation'],
        original_plan_sha256=manifest['provenance']['original_plan_sha256'],
        interpretation='Fixed inputs and recipe; bit-identical weights or a strength improvement are not guaranteed.')
    print(json.dumps(invocation, ensure_ascii=False, indent=2), flush=True)
    print(subprocess.list2cmdline(command) if sys.platform == 'win32' else shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    # Exclusive directory reservation prevents two launches sharing one result.
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error('Output directory was created by another process; refusing to continue')
    # Keep the trainer's reserved output directory empty: warm-start explicitly
    # rejects nonempty outputs. The adjacent record preserves interruption evidence.
    invocation.update(status='running', started_utc=timestamp())
    with record.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(invocation, ensure_ascii=False, indent=2) + '\n')
    code = 1
    try:
        code = subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode
        verify_manifest()
        invocation.update(status='completed' if code == 0 else 'failed', exit_code=code,
                          fixed_inputs_and_source_unchanged=True)
    except BaseException as exc:
        invocation.update(status='failed', error=str(exc), exit_code=code)
        raise
    finally:
        invocation['finished_utc'] = timestamp()
        record.write_text(json.dumps(invocation, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
