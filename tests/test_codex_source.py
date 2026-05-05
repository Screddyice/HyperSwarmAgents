"""CodexSource contract tests:
  - install() writes wrapper script with sentinel + 0755 mode
  - install() refuses if real binary missing or PATH ordering would not shadow
  - install() is idempotent (rewrites are content-stable)
  - uninstall() removes only the wrapper we wrote (sentinel-matched)
  - capture() summarises recent log files and never raises on missing dir
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from hyperswarm.sources.codex import WRAPPER_SENTINEL, CodexSource


def _stub_real_binary(tmp_path: Path) -> Path:
    real = tmp_path / "homebrew" / "bin" / "codex"
    real.parent.mkdir(parents=True)
    real.write_text("#!/bin/sh\necho real codex\n")
    real.chmod(0o755)
    return real


def test_install_writes_executable_wrapper(tmp_path: Path, monkeypatch):
    real = _stub_real_binary(tmp_path)
    wrapper = tmp_path / "local" / "bin" / "codex"
    monkeypatch.setenv(
        "PATH", f"{wrapper.parent}:{real.parent}", prepend=False
    )

    src = CodexSource({
        "binary": str(real),
        "wrapper_path": str(wrapper),
    })
    src.install()
    src.install()  # idempotent

    assert wrapper.exists()
    assert os.access(wrapper, os.X_OK)
    content = wrapper.read_text()
    assert WRAPPER_SENTINEL in content
    assert str(real) in content
    assert "capture --runtime codex" in content


def test_wrapper_bakes_absolute_hyperswarm_path(tmp_path: Path, monkeypatch):
    """The wrapper must resolve hyperswarm to an absolute path at install
    time so it works even if the user's PATH changes (e.g., a different
    shell rc, or codex is invoked from a script with a stripped PATH)."""
    real = _stub_real_binary(tmp_path)
    wrapper = tmp_path / "local" / "bin" / "codex"

    # Plant a fake hyperswarm binary that shutil.which will discover.
    fake_hs = tmp_path / "hs-bin" / "hyperswarm"
    fake_hs.parent.mkdir(parents=True)
    fake_hs.write_text("#!/bin/sh\nexit 0\n")
    fake_hs.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_hs.parent}:{wrapper.parent}:{real.parent}")

    src = CodexSource({"binary": str(real), "wrapper_path": str(wrapper)})
    src.install()

    content = wrapper.read_text()
    assert f'HYPERSWARM_BIN="{fake_hs}"' in content, (
        "wrapper must contain absolute path to hyperswarm, not bare name"
    )


def test_install_fails_when_real_binary_missing(tmp_path: Path, monkeypatch):
    wrapper = tmp_path / "wrapper" / "codex"
    monkeypatch.setenv("PATH", str(wrapper.parent))
    src = CodexSource({
        "binary": str(tmp_path / "does-not-exist"),
        "wrapper_path": str(wrapper),
    })
    with pytest.raises(RuntimeError, match="not found"):
        src.install()


def test_install_fails_when_path_would_not_shadow(tmp_path: Path, monkeypatch):
    real = _stub_real_binary(tmp_path)
    wrapper = tmp_path / "local" / "bin" / "codex"
    # Real before wrapper in PATH — wrapper will not intercept
    monkeypatch.setenv("PATH", f"{real.parent}:{wrapper.parent}")

    src = CodexSource({
        "binary": str(real),
        "wrapper_path": str(wrapper),
    })
    with pytest.raises(RuntimeError, match="not intercept"):
        src.install()


def test_uninstall_only_removes_managed_wrapper(tmp_path: Path, monkeypatch):
    real = _stub_real_binary(tmp_path)
    wrapper = tmp_path / "local" / "bin" / "codex"
    monkeypatch.setenv("PATH", f"{wrapper.parent}:{real.parent}")
    src = CodexSource({"binary": str(real), "wrapper_path": str(wrapper)})
    src.install()
    src.uninstall()
    assert not wrapper.exists()

    # User-written script with no sentinel must NOT be removed.
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text("#!/bin/sh\necho i am the user\n")
    src.uninstall()
    assert wrapper.exists()


def test_capture_handles_missing_log_dir(tmp_path: Path):
    src = CodexSource({
        "binary": "/usr/bin/true",
        "wrapper_path": str(tmp_path / "ignored"),
        "log_dir": str(tmp_path / "no-log-dir"),
    })
    entry = src.capture({"cwd": "/work", "since_ts": int(time.time()) - 60})
    assert entry.runtime == "codex"
    assert entry.cwd == "/work"
    assert "no codex log files" in entry.body.lower() or "no log activity" in entry.summary.lower()


def test_capture_summarises_recent_log(tmp_path: Path):
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    old = log_dir / "old.log"
    old.write_text("old content")
    old_mtime = time.time() - 7200
    os.utime(old, (old_mtime, old_mtime))

    new = log_dir / "session-1.log"
    new.write_text("new content during this session\nsome detail line\n")

    src = CodexSource({
        "binary": "/usr/bin/true",
        "wrapper_path": str(tmp_path / "ignored"),
        "log_dir": str(log_dir),
    })
    entry = src.capture({"cwd": "/work", "since_ts": int(time.time()) - 60})
    assert "session-1.log" in entry.summary
    assert "new content" in entry.body
    assert "old content" not in entry.body  # the old log was excluded
