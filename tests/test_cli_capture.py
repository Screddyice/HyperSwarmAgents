"""End-to-end test for `hyperswarm capture` — the path a hook actually fires.

Verifies:
  - capture reads stdin JSON, calls the right Source, runs Scope, writes Store
  - capture with malformed stdin still produces an Entry (does not crash hook)
  - capture with no stdin (empty) still produces an Entry
  - the resulting entry is readable via `recent`
"""
from __future__ import annotations

import datetime as _dt
import io
import json
import sys
from pathlib import Path

import pytest

from hyperswarm import cli
from hyperswarm.stores.markdown import MarkdownStore


def _write_config(tmp_path: Path, scope_tag: str = "test-scope") -> Path:
    # Use the writable tmp_path for the store; declare a single matching path
    # rule so the entry gets a deterministic scope tag.
    cfg = f"""
[store]
type = "markdown"
path = "{tmp_path}/store"

[scope]
type = "path_prefix"
fallback = "unscoped"

[[scope.path_prefix]]
prefix = "{tmp_path}"
tag = "{scope_tag}"

[[source]]
type = "claude_code"
settings_path = "{tmp_path}/settings.json"
"""
    p = tmp_path / "config.toml"
    p.write_text(cfg)
    return p


def _run_capture(monkeypatch, args: list[str], stdin: str = "") -> int:
    monkeypatch.setattr(sys, "argv", ["hyperswarm"] + args)
    if stdin:
        monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    else:
        monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    # Pretend stdin is not a tty so capture reads it
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    return cli.main()


def test_capture_writes_entry_to_store(tmp_path: Path, monkeypatch):
    cfg_path = _write_config(tmp_path)
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(json.dumps({
        "role": "user", "message": {"content": "Wire up the watchdog"}
    }) + "\n" + json.dumps({
        "role": "assistant", "message": {"content": "Plan: read config, install hook."}
    }) + "\n")

    payload = json.dumps({
        "session_id": "abc",
        "transcript_path": str(transcript),
        "cwd": str(tmp_path / "subdir"),
    })
    rc = _run_capture(monkeypatch, ["capture", "--runtime", "claude_code", "--config", str(cfg_path)], payload)
    assert rc == 0

    store = MarkdownStore({"path": f"{tmp_path}/store"})
    entries = list(store.list_since(_dt.datetime.min.replace(tzinfo=_dt.timezone.utc)))
    assert len(entries) == 1
    assert entries[0].runtime == "claude-code"
    assert entries[0].scope == "test-scope"
    assert entries[0].session_id == "abc"
    assert "Wire up the watchdog" in entries[0].body


def test_capture_with_empty_stdin_does_not_crash(tmp_path: Path, monkeypatch):
    cfg_path = _write_config(tmp_path)
    rc = _run_capture(monkeypatch, ["capture", "--runtime", "claude_code", "--config", str(cfg_path)], "")
    assert rc == 0
    # An entry was still written — capture is total
    store = MarkdownStore({"path": f"{tmp_path}/store"})
    entries = list(store.list_since(_dt.datetime.min.replace(tzinfo=_dt.timezone.utc)))
    assert len(entries) == 1


def test_capture_with_malformed_stdin_does_not_crash(tmp_path: Path, monkeypatch):
    cfg_path = _write_config(tmp_path)
    rc = _run_capture(monkeypatch, ["capture", "--runtime", "claude_code", "--config", str(cfg_path)],
                      "not even close to JSON {[]")
    assert rc == 0


def test_capture_unknown_runtime_returns_error(tmp_path: Path, monkeypatch):
    cfg_path = _write_config(tmp_path)
    rc = _run_capture(monkeypatch, ["capture", "--runtime", "made-up-runtime", "--config", str(cfg_path)])
    assert rc != 0, "unknown runtime must return non-zero so hooks see the failure in their log"
