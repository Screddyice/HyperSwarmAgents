"""ClaudeHarnessSource contract tests.

The source watches `.claude-harness/memory/procedural/{failures,successes}.json`,
`.claude-harness/memory/learned/*.json`, and `.claude-harness/features/archive.json`
across one or more project roots, emitting one HyperSwarm Entry per new
entry id seen.

Behaviour required:
  - install() snapshots all currently-present entry ids as "seen" so we don't
    replay history on first run; idempotent
  - capture() returns one Entry for an unseen id, then None when drained
  - capture() handles failures, successes, learned rules, and archived features
  - dedup is keyed by (harness_dir, source_file, entry_id) so two projects
    with overlapping ids don't collide
  - malformed JSON is tolerated (entry skipped, source keeps working)
  - missing roots are tolerated
  - runtime_name override flows through to Entry.runtime
  - cwd on the emitted Entry is the project dir (parent of .claude-harness/)
"""
from __future__ import annotations

import json
from pathlib import Path

from hyperswarm.sources.claude_harness import ClaudeHarnessSource


def _make(tmp_path: Path, **overrides) -> ClaudeHarnessSource:
    config = {
        "roots": [str(tmp_path / "projects")],
        "state_path": str(tmp_path / "state" / "claude_harness.json"),
        "max_depth": 4,
    }
    config.update(overrides)
    return ClaudeHarnessSource(config)


def _write_harness(project_dir: Path, *, failures=None, successes=None, learned=None, archive=None) -> None:
    """Build a .claude-harness/ tree at project_dir with the given entries."""
    harness = project_dir / ".claude-harness"
    proc = harness / "memory" / "procedural"
    learned_dir = harness / "memory" / "learned"
    features = harness / "features"
    proc.mkdir(parents=True, exist_ok=True)
    learned_dir.mkdir(parents=True, exist_ok=True)
    features.mkdir(parents=True, exist_ok=True)

    if failures is not None:
        (proc / "failures.json").write_text(json.dumps({"version": 3, "entries": failures}))
    if successes is not None:
        (proc / "successes.json").write_text(json.dumps({"version": 3, "entries": successes}))
    if learned is not None:
        # learned is a dict of filename -> entries list
        for fname, entries in learned.items():
            (learned_dir / fname).write_text(json.dumps({"version": 1, "entries": entries}))
    if archive is not None:
        (features / "archive.json").write_text(json.dumps({"version": 3, "features": archive, "fixes": []}))


def _failure(eid: str, *, feature="feature-001", approach="tried X", errors=None, root_cause="root", prevention="don't do that") -> dict:
    return {
        "id": eid,
        "timestamp": "2026-05-07T00:00:00Z",
        "feature": feature,
        "approach": approach,
        "errors": errors or ["error"],
        "rootCause": root_cause,
        "prevention": prevention,
    }


def _success(eid: str, *, feature="feature-001", approach="did X") -> dict:
    return {
        "id": eid,
        "timestamp": "2026-05-07T00:00:00Z",
        "feature": feature,
        "approach": approach,
    }


def _archive_feature(fid: str, *, name="thing", status="passing") -> dict:
    return {
        "id": fid,
        "name": name,
        "description": f"description for {fid}",
        "priority": 1,
        "status": status,
        "phase": None,
        "attempts": 1,
        "completedAt": "2026-05-07T01:00:00Z",
    }


# ----------------------------------------------------------------- install

def test_install_creates_state_file(tmp_path: Path):
    src = _make(tmp_path)
    src.install()
    assert Path(src.state_path).exists()


def test_install_snapshots_existing_ids_as_seen(tmp_path: Path):
    """Existing failures/successes/etc must NOT replay on first capture."""
    project = tmp_path / "projects" / "p1"
    project.mkdir(parents=True)
    _write_harness(project, failures=[_failure("fail-20260507-001")], successes=[_success("suc-20260507-001")])

    src = _make(tmp_path)
    src.install()

    assert src.capture({}) is None, "first capture must not replay pre-install history"


def test_install_is_idempotent(tmp_path: Path):
    project = tmp_path / "projects" / "p1"
    project.mkdir(parents=True)
    _write_harness(project, failures=[_failure("fail-20260507-001")])

    src = _make(tmp_path)
    src.install()
    state_1 = Path(src.state_path).read_text()
    src.install()
    state_2 = Path(src.state_path).read_text()
    assert state_1 == state_2


# ----------------------------------------------------------------- capture

def test_capture_returns_none_when_no_harness_dirs(tmp_path: Path):
    src = _make(tmp_path)
    src.install()
    assert src.capture({}) is None


def test_capture_returns_none_when_root_missing(tmp_path: Path):
    src = ClaudeHarnessSource({
        "roots": [str(tmp_path / "no-such-dir")],
        "state_path": str(tmp_path / "state.json"),
    })
    src.install()
    assert src.capture({}) is None


def test_capture_emits_failure_entry(tmp_path: Path):
    project = tmp_path / "projects" / "p1"
    project.mkdir(parents=True)

    src = _make(tmp_path)
    src.install()

    # Add a failure AFTER install — should be captured
    _write_harness(project, failures=[_failure("fail-20260507-001", approach="used the wrong API")])

    entry = src.capture({})
    assert entry is not None
    assert entry.runtime == "claude_harness"
    assert entry.cwd == str(project)
    assert "fail-20260507-001" in entry.session_id
    assert "FAILURE" in entry.summary or "failure" in entry.summary.lower()
    assert "used the wrong API" in entry.body


def test_capture_emits_success_entry(tmp_path: Path):
    project = tmp_path / "projects" / "p1"
    project.mkdir(parents=True)

    src = _make(tmp_path)
    src.install()
    _write_harness(project, successes=[_success("suc-20260507-001", approach="reused pattern X")])

    entry = src.capture({})
    assert entry is not None
    assert "suc-20260507-001" in entry.session_id
    assert "reused pattern X" in entry.body


def test_capture_emits_learned_entry(tmp_path: Path):
    project = tmp_path / "projects" / "p1"
    project.mkdir(parents=True)

    src = _make(tmp_path)
    src.install()
    _write_harness(project, learned={"rules.json": [{"id": "rule-001", "rule": "always X", "timestamp": "2026-05-07T00:00:00Z"}]})

    entry = src.capture({})
    assert entry is not None
    assert "rule-001" in entry.session_id
    assert "always X" in entry.body


def test_capture_emits_archive_entry(tmp_path: Path):
    project = tmp_path / "projects" / "p1"
    project.mkdir(parents=True)

    src = _make(tmp_path)
    src.install()
    _write_harness(project, archive=[_archive_feature("feature-001", name="dark mode", status="passing")])

    entry = src.capture({})
    assert entry is not None
    assert "feature-001" in entry.session_id
    assert "dark mode" in entry.body


def test_capture_drains_then_returns_none(tmp_path: Path):
    project = tmp_path / "projects" / "p1"
    project.mkdir(parents=True)

    src = _make(tmp_path)
    src.install()
    _write_harness(project, failures=[
        _failure("fail-20260507-001"),
        _failure("fail-20260507-002"),
        _failure("fail-20260507-003"),
    ])

    seen_ids = []
    while True:
        e = src.capture({})
        if e is None:
            break
        seen_ids.append(e.session_id)

    assert len(seen_ids) == 3, f"expected 3 entries, got {seen_ids}"
    # all three should have unique ids
    assert len(set(seen_ids)) == 3


def test_capture_dedupes_across_calls(tmp_path: Path):
    """Calling capture, then re-running install (or just calling capture again)
    must not re-emit the same entry."""
    project = tmp_path / "projects" / "p1"
    project.mkdir(parents=True)

    src = _make(tmp_path)
    src.install()
    _write_harness(project, failures=[_failure("fail-20260507-001")])

    e1 = src.capture({})
    assert e1 is not None
    e2 = src.capture({})
    assert e2 is None, "second capture should not re-emit a seen id"


def test_capture_walks_multiple_projects(tmp_path: Path):
    p1 = tmp_path / "projects" / "p1"
    p1.mkdir(parents=True)
    p2 = tmp_path / "projects" / "org-a" / "p2"
    p2.mkdir(parents=True)

    src = _make(tmp_path)
    src.install()
    _write_harness(p1, failures=[_failure("fail-20260507-001", approach="A")])
    _write_harness(p2, failures=[_failure("fail-20260507-001", approach="B")])  # same id, different project

    seen_cwds = []
    while True:
        e = src.capture({})
        if e is None:
            break
        seen_cwds.append(e.cwd)

    # Both projects should emit despite sharing an entry id
    assert str(p1) in seen_cwds
    assert str(p2) in seen_cwds


def test_capture_tolerates_malformed_json(tmp_path: Path):
    """A malformed failures.json must not crash the capture loop or block
    other valid sources from being read."""
    project = tmp_path / "projects" / "p1"
    project.mkdir(parents=True)

    src = _make(tmp_path)
    src.install()

    # Build harness with broken failures.json but valid successes.json
    harness = project / ".claude-harness"
    proc = harness / "memory" / "procedural"
    proc.mkdir(parents=True)
    (proc / "failures.json").write_text("{not valid json")
    (proc / "successes.json").write_text(json.dumps({
        "version": 3,
        "entries": [_success("suc-20260507-001", approach="worked")],
    }))

    # Should still capture the success
    seen = []
    while True:
        e = src.capture({})
        if e is None:
            break
        seen.append(e.session_id)

    assert any("suc-20260507-001" in s for s in seen)


def test_runtime_name_override(tmp_path: Path):
    project = tmp_path / "projects" / "p1"
    project.mkdir(parents=True)

    src = ClaudeHarnessSource({
        "roots": [str(tmp_path / "projects")],
        "state_path": str(tmp_path / "state.json"),
        "runtime_name": "claude-harness-mac",
    })
    src.install()
    _write_harness(project, failures=[_failure("fail-20260507-001")])

    e = src.capture({})
    assert e is not None
    assert e.runtime == "claude-harness-mac"
