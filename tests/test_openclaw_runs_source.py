"""OpenClawRunsSource contract tests."""
from __future__ import annotations

import json
from pathlib import Path

from hyperswarm.sources.openclaw_runs import OpenClawRunsSource


def _make(tmp_path: Path) -> OpenClawRunsSource:
    return OpenClawRunsSource({
        "runs_dir": str(tmp_path / "runs"),
        "state_path": str(tmp_path / "state.json"),
    })


def _write_jsonl(tmp_path: Path, name: str, records: list[dict]) -> Path:
    runs = tmp_path / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    f = runs / name
    f.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return f


def test_install_seeds_cursor_at_existing_line_counts(tmp_path: Path):
    _write_jsonl(tmp_path, "job-a.jsonl", [{"jobId": "a", "ts": 1, "action": "finished", "status": "ok"}])
    _write_jsonl(tmp_path, "job-b.jsonl", [
        {"jobId": "b", "ts": 1, "action": "finished", "status": "ok"},
        {"jobId": "b", "ts": 2, "action": "finished", "status": "error", "error": "boom"},
    ])
    src = _make(tmp_path)
    src.install()
    state = json.loads(Path(src.state_path).read_text())
    assert state == {"job-a.jsonl": 1, "job-b.jsonl": 2}


def test_install_does_not_replay_history(tmp_path: Path):
    _write_jsonl(tmp_path, "job.jsonl", [
        {"jobId": "x", "ts": 1, "action": "finished", "status": "ok", "summary": "old run"}
    ])
    src = _make(tmp_path)
    src.install()
    assert src.capture({}) is None


def test_install_idempotent_preserves_cursor(tmp_path: Path):
    _write_jsonl(tmp_path, "job.jsonl", [{"jobId": "x", "ts": 1, "action": "started", "status": "ok"}])
    src = _make(tmp_path)
    src.install()
    src.install()
    state = json.loads(Path(src.state_path).read_text())
    assert state == {"job.jsonl": 1}


def test_capture_returns_one_entry_per_new_line(tmp_path: Path):
    src = _make(tmp_path)
    src.install()  # cursor at empty
    f = _write_jsonl(tmp_path, "job.jsonl", [
        {"jobId": "j1", "sessionId": "s1", "ts": 1700000000000, "action": "finished",
         "status": "ok", "summary": "Sent morning report", "durationMs": 4231,
         "model": "openai-codex/gpt-5.4", "provider": "openai"},
        {"jobId": "j1", "sessionId": "s2", "ts": 1700000060000, "action": "finished",
         "status": "error", "error": "API rate limit reached"},
    ])

    e1 = src.capture({})
    assert e1 is not None
    assert e1.runtime == "openclaw_runs"
    assert "Sent morning report" in e1.summary
    assert "j1" in e1.body
    assert "openai-codex/gpt-5.4" in e1.body

    e2 = src.capture({})
    assert e2 is not None
    assert "API rate limit" in e2.summary
    assert "API rate limit" in e2.body  # error included in body

    e3 = src.capture({})
    assert e3 is None


def test_capture_advances_cursor_across_files(tmp_path: Path):
    src = _make(tmp_path)
    src.install()
    _write_jsonl(tmp_path, "job-a.jsonl", [
        {"jobId": "a", "ts": 1, "action": "finished", "status": "ok", "summary": "A1"},
    ])
    _write_jsonl(tmp_path, "job-b.jsonl", [
        {"jobId": "b", "ts": 1, "action": "finished", "status": "ok", "summary": "B1"},
    ])
    summaries = []
    for _ in range(5):
        e = src.capture({})
        if e is None:
            break
        summaries.append(e.summary)
    assert "A1" in summaries
    assert "B1" in summaries


def test_capture_with_missing_runs_dir_returns_none(tmp_path: Path):
    src = OpenClawRunsSource({
        "runs_dir": str(tmp_path / "no-such"),
        "state_path": str(tmp_path / "state.json"),
    })
    src.install()
    assert src.capture({}) is None


def test_runtime_name_override(tmp_path: Path):
    src = OpenClawRunsSource({
        "runs_dir": str(tmp_path / "runs"),
        "state_path": str(tmp_path / "state.json"),
        "runtime_name": "openclaw-runs-neb",
    })
    src.install()
    _write_jsonl(tmp_path, "job.jsonl", [
        {"jobId": "x", "ts": 1700000000000, "action": "finished", "status": "ok", "summary": "ran"}
    ])
    e = src.capture({})
    assert e is not None
    assert e.runtime == "openclaw-runs-neb"


def test_malformed_line_does_not_crash(tmp_path: Path):
    src = _make(tmp_path)
    src.install()
    runs = tmp_path / "runs"
    runs.mkdir(parents=True)
    f = runs / "broken.jsonl"
    f.write_text("not json\n{still bad\n")
    e = src.capture({})
    # Returns None for the malformed line — caller can try again on next tick
    assert e is None


def test_uninstall_clears_state(tmp_path: Path):
    src = _make(tmp_path)
    src.install()
    assert Path(src.state_path).exists()
    src.uninstall()
    assert not Path(src.state_path).exists()
