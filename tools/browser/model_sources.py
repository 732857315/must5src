"""Shared checkpoint identity for browser export, release checks and parity.

Relative paths always mean repository-relative, never process-CWD-relative.
Only a missing checkpoint_path in an old manifest invokes the legacy default;
an explicit malformed/missing path or a wrong SHA never falls back to v3.
This module is standard-library only and never loads a model.
"""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINTS = {
    'opponent': 'training_runs/unet_curriculum_v2/opponent.pt',
    'play': 'training_runs/unet_curriculum_v2/play.pt',
    'global': 'training_runs/global_v3/global.pt',
}
_DEFAULT = object()


def resolve_checkpoint(role, path=_DEFAULT, *, root=ROOT):
    """Resolve one real source file; explicit external absolute paths are valid."""
    if role not in DEFAULT_CHECKPOINTS:
        raise ValueError('Unknown browser model role: ' + str(role))
    if path is _DEFAULT:
        path = DEFAULT_CHECKPOINTS[role]
    if not isinstance(path, (str, Path)) or not str(path).strip():
        raise ValueError('Invalid checkpoint_path for ' + role)
    root = Path(root).resolve()
    candidate = Path(path)
    candidate = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if not candidate.is_file():
        raise ValueError('Source checkpoint missing or not a file: ' + str(candidate))
    return candidate


def checkpoint_metadata(role, path=_DEFAULT, *, root=ROOT):
    """Record the actual source path and bytes before export starts."""
    root = Path(root).resolve()
    resolved = resolve_checkpoint(role, path, root=root)
    recorded = resolved.relative_to(root) if resolved.is_relative_to(root) else resolved
    return dict(checkpoint_path=recorded.as_posix(),
                checkpoint_sha256=hashlib.sha256(resolved.read_bytes()).hexdigest())


def verified_checkpoint_paths(manifest, *, root=ROOT):
    """Check ALL role identities before callers load any reference model.

    Return role -> resolved Path. An old manifest without path metadata remains
    supported, but its recorded SHA must match the legacy source file.
    """
    if not isinstance(manifest, dict) or not isinstance(manifest.get('models'), dict):
        raise ValueError('Browser manifest needs a models mapping')
    result = {}
    for role in DEFAULT_CHECKPOINTS:
        metadata = manifest['models'].get(role)
        if not isinstance(metadata, dict):
            raise ValueError('Missing source model metadata: ' + role)
        expected = metadata.get('checkpoint_sha256')
        if (not isinstance(expected, str) or len(expected) != 64
                or any(c not in '0123456789abcdefABCDEF' for c in expected)):
            raise ValueError('Invalid source checkpoint SHA256: ' + role)
        path = resolve_checkpoint(role, metadata.get('checkpoint_path', _DEFAULT), root=root)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected.lower():
            raise ValueError('Source checkpoint SHA256 mismatch: ' + role)
        result[role] = path
    return result
