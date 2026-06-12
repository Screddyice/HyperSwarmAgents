"""ClaudeMemSessionSource contract tests.

Covers the session-level, significance-gated reflector:

  - install() merges a SessionEnd hook idempotently, preserves other hooks
  - capture() reads claude-mem.db READ-ONLY (no mutation, query_only enforced)
  - capture() returns exactly ONE Entry for a session WITH a course-change/lesson
  - capture() returns None for a ROUTINE session (structural pre-filter, no LLM)
  - the structural pre-filter short-circuits routine sessions without an LLM call
  - the LLM gate qualifies course_change / missed_pr / lesson fixtures
  - session_id on the emitted Entry == claude-mem memory_session_id (idempotency key)
  - capture() never raises on missing db / missing session

A real claude-mem.db schema subset is created in tmp for each test; the LLM
gate is injected via the `gate_fn` config seam so tests never shell out to codex.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from hyperswarm.core.entry import Entry
from hyperswarm.sources.claude_mem_session import (
    ClaudeMemSessionSource,
    _parse_gate_json,
    _GATE_SYSTEM_PROMPT,
)


# --------------------------------------------------------------------------- #
# DB fixture — minimal subset of the real claude-mem schema
# --------------------------------------------------------------------------- #

def _make_db(tmp_path: Path) -> Path:
    db = tmp_path / "claude-mem.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE sdk_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content_session_id TEXT UNIQUE NOT NULL,
            memory_session_id TEXT UNIQUE,
            project TEXT NOT NULL,
            platform_source TEXT NOT NULL DEFAULT 'claude',
            user_prompt TEXT,
            started_at TEXT NOT NULL DEFAULT '',
            started_at_epoch INTEGER NOT NULL DEFAULT 0,
            completed_at TEXT,
            completed_at_epoch INTEGER,
            status TEXT NOT NULL DEFAULT 'active'
        );
        CREATE TABLE session_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_session_id TEXT NOT NULL,
            project TEXT NOT NULL,
            request TEXT,
            investigated TEXT,
            learned TEXT,
            completed TEXT,
            next_steps TEXT,
            files_read TEXT,
            files_edited TEXT,
            notes TEXT,
            prompt_number INTEGER,
            discovery_tokens INTEGER DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT '',
            created_at_epoch INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_session_id TEXT NOT NULL,
            project TEXT NOT NULL,
            text TEXT,
            type TEXT NOT NULL DEFAULT 'note',
            content_hash TEXT,
            created_at TEXT NOT NULL DEFAULT '',
            created_at_epoch INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    conn.commit()
    conn.close()
    return db


def _insert_session(
    db: Path,
    *,
    content_session_id: str,
    memory_session_id: str | None,
    status: str = "completed",
    project: str = "projects",
    user_prompt: str = "do the thing",
    summary: dict | None = None,
    n_observations: int = 3,
):
    conn = sqlite3.connect(str(db))
    conn.execute(
        "INSERT INTO sdk_sessions (content_session_id, memory_session_id, project, user_prompt, status) "
        "VALUES (?, ?, ?, ?, ?)",
        (content_session_id, memory_session_id, project, user_prompt, status),
    )
    if memory_session_id and summary is not None:
        conn.execute(
            "INSERT INTO session_summaries "
            "(memory_session_id, project, request, investigated, learned, completed, next_steps, "
            " files_read, files_edited, notes, prompt_number, created_at_epoch) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                memory_session_id,
                project,
                summary.get("request", ""),
                summary.get("investigated", ""),
                summary.get("learned", ""),
                summary.get("completed", ""),
                summary.get("next_steps", ""),
                summary.get("files_read", ""),
                summary.get("files_edited", ""),
                summary.get("notes", ""),
                summary.get("prompt_number", 1),
                summary.get("created_at_epoch", 1000),
            ),
        )
        for i in range(n_observations):
            conn.execute(
                "INSERT INTO observations (memory_session_id, project, text, content_hash, created_at_epoch) "
                "VALUES (?, ?, ?, ?, ?)",
                (memory_session_id, project, f"turn {i}", f"h{i}", 1000 + i),
            )
    conn.commit()
    conn.close()


# Gate seams — deterministic, no codex.
def _qualifying_gate(trigger: str):
    def _fn(_prompt: str) -> str:
        return json.dumps(
            {
                "qualifies": True,
                "trigger": trigger,
                "headline": f"{trigger} headline",
                "lesson_body": f"the durable {trigger} lesson",
                "reason": "fixture",
            }
        )
    return _fn


def _never_called_gate(_prompt: str) -> str:  # pragma: no cover - asserted never invoked
    raise AssertionError("LLM gate should NOT be called for a routine session")


def _declining_gate(_prompt: str) -> str:
    return json.dumps(
        {"qualifies": False, "trigger": None, "headline": "", "lesson_body": "", "reason": "routine"}
    )


_SIGNIFICANT_SUMMARY = {
    "request": "Wire up the X feature end to end",
    "investigated": "Tried approach A, it didn't work",
    "learned": "Approach A fails because of the WAL lock; switched to mode=ro which is the real fix.",
    "completed": "Switched to read-only connect",
    "next_steps": "Deploy to hostinger and verify recall separation",
    "notes": "Gotcha: query_only pragma is required to prevent accidental writes.",
}

_ROUTINE_SUMMARY = {
    "request": "Add a comment to the readme",
    "investigated": "",
    "learned": "",
    "completed": "Added the comment",
    "next_steps": "No active work in progress.",
    "notes": "",
}

# Not lesson-worthy (no pivot, no gotcha) but ends with real unfinished work —
# the canonical "where Claude Code left off" handoff state.
_LEFTOFF_SUMMARY = {
    "request": "Add the Outreach tab to the client panel",
    "investigated": "Read the existing Research tab wiring",
    "learned": "Existing tab registry pattern was reused as-is",
    "completed": "Tab scaffold and API client committed on feat/be-outreach-tab",
    "next_steps": "Wire the send-queue endpoint, add bun:test coverage, then open the PR for review",
    "notes": "",
    "files_edited": "src/panels/outreach.tsx, src/api/sendqueue.ts",
}


# --------------------------------------------------------------------------- #
# install()
# --------------------------------------------------------------------------- #

def test_install_idempotent_session_end(tmp_path: Path):
    settings = tmp_path / "settings.json"
    src = ClaudeMemSessionSource({"settings_path": str(settings)})
    src.install()
    src.install()
    src.install()

    data = json.loads(settings.read_text())
    se = data["hooks"]["SessionEnd"]
    cmds = [h["command"] for entry in se for h in entry["hooks"]]
    ours = [c for c in cmds if "capture --runtime claude_mem_session" in c]
    assert len(ours) == 1, f"expected 1 hook, got {len(ours)}"


def test_install_preserves_other_hooks(tmp_path: Path):
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "theme": "dark",
                "hooks": {
                    "Stop": [{"matcher": "Y", "hooks": [{"type": "command", "command": "echo stop"}]}]
                },
            }
        )
    )
    src = ClaudeMemSessionSource({"settings_path": str(settings)})
    src.install()
    data = json.loads(settings.read_text())
    assert data["theme"] == "dark"
    stop_cmds = [h["command"] for e in data["hooks"]["Stop"] for h in e["hooks"]]
    assert "echo stop" in stop_cmds
    se_cmds = [h["command"] for e in data["hooks"]["SessionEnd"] for h in e["hooks"]]
    assert any("capture --runtime claude_mem_session" in c for c in se_cmds)


def test_uninstall_is_noop_when_absent(tmp_path: Path):
    settings = tmp_path / "settings.json"
    src = ClaudeMemSessionSource({"settings_path": str(settings)})
    src.uninstall()  # must not raise even with no file


# --------------------------------------------------------------------------- #
# capture() — qualifying vs routine
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("trigger", ["course_change", "missed_pr", "lesson"])
def test_capture_emits_one_entry_for_significant_session(tmp_path: Path, trigger: str):
    db = _make_db(tmp_path)
    _insert_session(
        db,
        content_session_id="cs-1",
        memory_session_id="mem-1",
        summary=_SIGNIFICANT_SUMMARY,
    )
    src = ClaudeMemSessionSource(
        {
            "settings_path": str(tmp_path / "settings.json"),
            "db_path": str(db),
            "gate_fn": _qualifying_gate(trigger),
        }
    )
    entry = src.capture({"session_id": "cs-1", "cwd": "/Users/x/projects/Screddyice/foo"})
    assert isinstance(entry, Entry)
    assert entry.runtime == "claude-mem-session"
    assert entry.session_id == "mem-1"  # stable idempotency key = memory_session_id
    assert entry.summary  # headline present
    assert "## Trigger" in entry.body and trigger in entry.body
    assert "## Lesson" in entry.body
    # cwd preserved (so git_remote scope can resolve), project carried separately
    assert entry.cwd == "/Users/x/projects/Screddyice/foo"
    assert entry.project == "projects"


def test_capture_returns_none_for_routine_session(tmp_path: Path):
    db = _make_db(tmp_path)
    _insert_session(
        db,
        content_session_id="cs-routine",
        memory_session_id="mem-routine",
        summary=_ROUTINE_SUMMARY,
    )
    # Inject a gate that MUST NOT be called — proves the structural pre-filter
    # short-circuits routine sessions without spending an LLM call.
    src = ClaudeMemSessionSource(
        {
            "settings_path": str(tmp_path / "settings.json"),
            "db_path": str(db),
            "gate_fn": _never_called_gate,
        }
    )
    entry = src.capture({"session_id": "cs-routine", "cwd": "/tmp"})
    assert entry is None


def test_capture_returns_none_when_gate_declines_and_no_handoff(tmp_path: Path):
    """Gate declines AND next_steps is boilerplate -> nothing is written."""
    db = _make_db(tmp_path)
    summary = dict(_SIGNIFICANT_SUMMARY, next_steps="No active work in progress.")
    _insert_session(
        db,
        content_session_id="cs-decline",
        memory_session_id="mem-decline",
        summary=summary,  # has teaching content -> reaches LLM gate
    )
    src = ClaudeMemSessionSource(
        {
            "settings_path": str(tmp_path / "settings.json"),
            "db_path": str(db),
            "gate_fn": _declining_gate,
        }
    )
    assert src.capture({"session_id": "cs-decline", "cwd": "/tmp"}) is None


# --------------------------------------------------------------------------- #
# capture() — leftoff handoff fallback
# --------------------------------------------------------------------------- #

def test_gate_decline_with_substantive_next_steps_emits_leftoff(tmp_path: Path):
    """Not lesson-worthy but real unfinished work -> ONE leftoff handoff Entry."""
    db = _make_db(tmp_path)
    _insert_session(
        db,
        content_session_id="cs-leftoff",
        memory_session_id="mem-leftoff",
        project="hyperscale",
        summary=_LEFTOFF_SUMMARY,
    )
    src = ClaudeMemSessionSource(
        {
            "settings_path": str(tmp_path / "settings.json"),
            "db_path": str(db),
            "gate_fn": _declining_gate,
        }
    )
    cwd = "/Users/x/projects/tmn/teamnebula.ai/hyperscale"
    entry = src.capture({"session_id": "cs-leftoff", "cwd": cwd})
    assert isinstance(entry, Entry)
    assert entry.runtime == "claude-mem-session"
    assert entry.session_id == "mem-leftoff"  # same idempotency key as lessons
    assert entry.summary.startswith("Left off:")
    assert "## Trigger" in entry.body and "leftoff" in entry.body
    assert "## Left off" in entry.body
    # Hermes prefetch scores keyword overlap against entry.body ONLY — the
    # project name, cwd, and next steps must appear in the body verbatim.
    assert "hyperscale" in entry.body
    assert cwd in entry.body
    assert _LEFTOFF_SUMMARY["next_steps"] in entry.body
    assert entry.cwd == cwd
    assert entry.project == "hyperscale"


def test_gate_error_with_next_steps_still_emits_leftoff(tmp_path: Path):
    """Codex down must not lose the handoff: gate error -> leftoff Entry."""
    db = _make_db(tmp_path)
    _insert_session(
        db,
        content_session_id="cs-gate-err",
        memory_session_id="mem-gate-err",
        summary=_LEFTOFF_SUMMARY,
    )

    def _raising_gate(_p: str) -> str:
        raise RuntimeError("codex unreachable")

    src = ClaudeMemSessionSource(
        {
            "settings_path": str(tmp_path / "settings.json"),
            "db_path": str(db),
            "gate_fn": _raising_gate,
        }
    )
    entry = src.capture({"session_id": "cs-gate-err", "cwd": "/tmp"})
    assert isinstance(entry, Entry)
    assert "leftoff" in entry.body


def test_qualifying_lesson_takes_precedence_over_leftoff(tmp_path: Path):
    """A qualifying session emits the LESSON entry (which already carries
    next_steps in its ## Session block) — never a second leftoff entry."""
    db = _make_db(tmp_path)
    _insert_session(
        db,
        content_session_id="cs-both",
        memory_session_id="mem-both",
        summary=_SIGNIFICANT_SUMMARY,  # substantive next_steps AND a lesson
    )
    src = ClaudeMemSessionSource(
        {
            "settings_path": str(tmp_path / "settings.json"),
            "db_path": str(db),
            "gate_fn": _qualifying_gate("lesson"),
        }
    )
    entry = src.capture({"session_id": "cs-both", "cwd": "/tmp"})
    assert isinstance(entry, Entry)
    assert "## Lesson" in entry.body
    assert "## Left off" not in entry.body
    assert _SIGNIFICANT_SUMMARY["next_steps"] in entry.body  # rides ## Session


@pytest.mark.parametrize(
    "next_steps",
    [
        "",
        "None",
        "n/a",
        "Done.",
        "No active work in progress.",
        "No further action needed at this time.",
        "short",
    ],
)
def test_boilerplate_next_steps_do_not_emit_leftoff(tmp_path: Path, next_steps: str):
    db = _make_db(tmp_path)
    summary = dict(_LEFTOFF_SUMMARY, next_steps=next_steps)
    _insert_session(
        db,
        content_session_id=f"cs-bp-{abs(hash(next_steps))}",
        memory_session_id=f"mem-bp-{abs(hash(next_steps))}",
        summary=summary,
    )
    src = ClaudeMemSessionSource(
        {
            "settings_path": str(tmp_path / "settings.json"),
            "db_path": str(db),
            "gate_fn": _declining_gate,
        }
    )
    assert (
        src.capture({"session_id": f"cs-bp-{abs(hash(next_steps))}", "cwd": "/tmp"})
        is None
    )


# --------------------------------------------------------------------------- #
# capture() — resilience / missing data
# --------------------------------------------------------------------------- #

def test_capture_none_when_db_missing(tmp_path: Path):
    src = ClaudeMemSessionSource(
        {"settings_path": str(tmp_path / "s.json"), "db_path": str(tmp_path / "nope.db")}
    )
    assert src.capture({"session_id": "cs-1", "cwd": "/tmp"}) is None


def test_capture_none_when_session_unknown(tmp_path: Path):
    db = _make_db(tmp_path)
    src = ClaudeMemSessionSource(
        {"settings_path": str(tmp_path / "s.json"), "db_path": str(db), "gate_fn": _never_called_gate}
    )
    assert src.capture({"session_id": "missing", "cwd": "/tmp"}) is None


def test_capture_none_when_no_memory_session_id(tmp_path: Path):
    """Active session not yet completed → claude-mem hasn't minted a memory id."""
    db = _make_db(tmp_path)
    _insert_session(
        db,
        content_session_id="cs-active",
        memory_session_id=None,
        status="active",
        summary=None,
    )
    src = ClaudeMemSessionSource(
        {"settings_path": str(tmp_path / "s.json"), "db_path": str(db), "gate_fn": _never_called_gate}
    )
    assert src.capture({"session_id": "cs-active", "cwd": "/tmp"}) is None


def test_capture_none_when_no_session_id_in_payload(tmp_path: Path):
    db = _make_db(tmp_path)
    src = ClaudeMemSessionSource(
        {"settings_path": str(tmp_path / "s.json"), "db_path": str(db), "gate_fn": _never_called_gate}
    )
    assert src.capture({"cwd": "/tmp"}) is None


# --------------------------------------------------------------------------- #
# Read-only safety — claude-mem must never be mutated (North Star)
# --------------------------------------------------------------------------- #

def test_db_opened_read_only_rejects_writes(tmp_path: Path):
    db = _make_db(tmp_path)
    _insert_session(
        db, content_session_id="cs-ro", memory_session_id="mem-ro", summary=_SIGNIFICANT_SUMMARY
    )
    src = ClaudeMemSessionSource({"db_path": str(db)})
    conn = src._connect_ro()
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO observations (memory_session_id, project, text) VALUES ('x','y','z')")
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("UPDATE session_summaries SET learned = 'tampered'")
    finally:
        conn.close()


def test_capture_does_not_mutate_db(tmp_path: Path):
    db = _make_db(tmp_path)
    _insert_session(
        db, content_session_id="cs-ro2", memory_session_id="mem-ro2", summary=_SIGNIFICANT_SUMMARY
    )
    before = db.read_bytes()
    src = ClaudeMemSessionSource(
        {"db_path": str(db), "gate_fn": _qualifying_gate("lesson"), "settings_path": str(tmp_path / "s.json")}
    )
    src.capture({"session_id": "cs-ro2", "cwd": "/tmp"})
    # Read-only access never writes back to the main db file.
    assert db.read_bytes() == before


def test_capture_idempotent_session_id_stable(tmp_path: Path):
    """Two capture() runs on the same session yield the SAME session_id, so the
    corpus (keyed on source_entry_path + session_id) dedups downstream."""
    db = _make_db(tmp_path)
    _insert_session(
        db, content_session_id="cs-id", memory_session_id="mem-id", summary=_SIGNIFICANT_SUMMARY
    )
    src = ClaudeMemSessionSource(
        {"db_path": str(db), "gate_fn": _qualifying_gate("lesson"), "settings_path": str(tmp_path / "s.json")}
    )
    e1 = src.capture({"session_id": "cs-id", "cwd": "/tmp"})
    e2 = src.capture({"session_id": "cs-id", "cwd": "/tmp"})
    assert e1.session_id == e2.session_id == "mem-id"


# --------------------------------------------------------------------------- #
# Gate JSON parsing
# --------------------------------------------------------------------------- #

def test_parse_gate_json_qualifying():
    out = _parse_gate_json('{"qualifies": true, "trigger": "lesson", "headline": "h", "lesson_body": "b", "reason": "r"}')
    assert out["qualifies"] is True
    assert out["trigger"] == "lesson"
    assert out["headline"] == "h"


def test_parse_gate_json_invalid_trigger_coerced_to_not_qualifying():
    out = _parse_gate_json('{"qualifies": true, "trigger": "banana", "headline": "h"}')
    assert out["qualifies"] is False


def test_parse_gate_json_garbage_is_safe():
    assert _parse_gate_json("not json at all")["qualifies"] is False
    assert _parse_gate_json("")["qualifies"] is False


def test_parse_gate_json_embedded_in_prose():
    raw = 'Sure! Here is the result:\n{"qualifies": false, "trigger": null, "headline": "", "lesson_body": "", "reason": "routine"}\nDone.'
    assert _parse_gate_json(raw)["qualifies"] is False


def test_gate_prompt_lists_three_triggers():
    for t in ("course_change", "missed_pr", "lesson"):
        assert t in _GATE_SYSTEM_PROMPT
