"""Publish result directories without overwriting previous user data."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import tempfile


def require_empty_output(output: Path | str) -> Path:
    """Return a destination that is absent or an empty, real directory."""
    path = Path(output)
    if path.is_symlink() or (path.exists() and (not path.is_dir() or any(path.iterdir()))):
        raise ValueError("Choose a new or empty output directory")
    return path


@contextmanager
def staged_output(output: Path | str):
    """Stage on the destination filesystem, then publish a complete result.

    A failure during export deletes only the temporary staging directory.
    Existing non-empty output is never cleared or replaced. A second check
    before publication protects against a destination populated during work.
    """
    target = require_empty_output(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".shardelix-", dir=target.parent) as temporary:
        stage = Path(temporary) / "result"
        stage.mkdir()
        yield stage
        require_empty_output(target)
        # Windows cannot rename onto an existing directory. rmdir succeeds
        # only when it is still empty; never recursively remove user data.
        if target.exists():
            target.rmdir()
        stage.rename(target)
