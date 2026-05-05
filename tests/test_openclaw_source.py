"""OpenClawSource contract tests:
  - install() seeds cursor at now() so we don't replay history
  - install() is idempotent (existing state is preserved)
  - capture() returns None when no new files
  - capture() processes oldest unseen first and advances cursor monotonically
  - capture() handles missing watch_dir cleanly
  - uninstall() removes state and lets reinstall start fresh
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from hyperswarm.sources.openclaw import OpenClawSource


def _make(tmp_path: Path, watch_subdir: str = "entries", state_subdir: str = "state") -> OpenClawSource:
    return OpenClawSource({
        "watch_dir": str(tmp_path / watch_subdir),
        "state_path": str(tmp_path / state_subdir / "openclaw.json"),
    })


def test_install_seeds_cursor_at_now(tmp_path: Path):
    src = _make(tmp_path)
    before = time.time()
    src.install()
    after = time.time()

    state = json.loads(Path(src.state_path).read_text())
    assert before <= state["cursor_mtime"] <= after


def test_install_does_not_replay_history(tmp_path: Path):
    """Files older than the cursor that existed before install must be skipped
    on the first capture call. This is the regression that protects us from
    flooding the store with thousands of historical entries on first run."""
    watch_dir = tmp_path / "entries"
    watch_dir.mkdir()
    old = watch_dir / "old.md"
    old.write_text("old entry")
    old_mtime = time.time() - 86400
    os.utime(old, (old_mtime, old_mtime))

    src = _make(tmp_path)
    src.install()
    entry = src.capture({})
    assert entry is None, "first capture must not replay pre-install history"


def test_install_idempotent_preserves_cursor(tmp_path: Path):
    src = _make(tmp_path)
    src.install()
    cursor_1 = json.loads(Path(src.state_path).read_text())["cursor_mtime"]
    time.sleep(0.05)
    src.install()
    cursor_2 = json.loads(Path(src.state_path).read_text())["cursor_mtime"]
    assert cursor_1 == cursor_2


def test_capture_returns_none_when_nothing_new(tmp_path: Path):
    src = _make(tmp_path)
    src.install()
    assert src.capture({}) is None


def test_capture_processes_oldest_first_and_advances_cursor(tmp_path: Path):
    src = _make(tmp_path)
    src.install()

    watch_dir = Path(src.watch_dir)
    watch_dir.mkdir(parents=True, exist_ok=True)

    # Three entries written after install, with explicit increasing mtimes.
    for i, name in enumerate(["a.md", "b.md", "c.md"]):
        f = watch_dir / name
        f.write_text(f"# entry {name}\nbody {i}")
        t = time.time() + i  # ensure strict ordering
        os.utime(f, (t, t))

    seen_summaries = []
    while True:
        e = src.capture({})
        if e is None:
            break
        seen_summaries.append(e.summary)
    # We saw all three, in age order
    assert len(seen_summaries) == 3
    assert "entry a.md" in seen_summaries[0]
    assert "entry c.md" in seen_summaries[2]


def test_capture_with_missing_watch_dir_returns_none(tmp_path: Path):
    src = OpenClawSource({
        "watch_dir": str(tmp_path / "no-such-dir"),
        "state_path": str(tmp_path / "state.json"),
    })
    src.install()
    assert src.capture({}) is None


def test_runtime_name_override(tmp_path: Path):
    src = OpenClawSource({
        "watch_dir": str(tmp_path / "entries"),
        "state_path": str(tmp_path / "state.json"),
        "runtime_name": "openclaw-neb",
    })
    src.install()
    watch = Path(src.watch_dir)
    watch.mkdir()
    f = watch / "x.md"
    f.write_text("body")
    t = time.time() + 1
    os.utime(f, (t, t))
    e = src.capture({})
    assert e is not None
    assert e.runtime == "openclaw-neb"


def test_uninstall_removes_state(tmp_path: Path):
    src = _make(tmp_path)
    src.install()
    assert Path(src.state_path).exists()
    src.uninstall()
    assert not Path(src.state_path).exists()
