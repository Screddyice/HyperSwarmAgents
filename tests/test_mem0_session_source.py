"""Mem0SessionSource contract tests — Mem0 Platform session distiller.

Gate is injected via gate_fn; memories via memories_fn so tests never hit
the live API or codex.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from hyperswarm.core.entry import Entry
from hyperswarm.sources import SOURCE_REGISTRY
from hyperswarm.sources.mem0_session import Mem0SessionSource


def _memories(session_id: str) -> list[dict]:
    now = datetime.now(timezone.utc).isoformat()
    return [
        {
            "id": "m1",
            "memory": "We switched cross-session agent memory from claude-mem to Mem0 Platform.",
            "metadata": {"type": "session_summary", "session_id": session_id},
            "created_at": now,
        },
        {
            "id": "m2",
            "memory": "HyperSwarm feed must distill Mem0 session facts, not read claude-mem.db.",
            "metadata": {"session_id": session_id},
            "created_at": now,
        },
        {
            "id": "m3",
            "memory": "Next steps: wire mem0_session capture and push to Hostinger.",
            "metadata": {"session_id": session_id},
            "created_at": now,
        },
    ]


def _gate_qualifies(trigger: str = "lesson"):
    def _fn(_prompt: str) -> str:
        return json.dumps(
            {
                "qualifies": True,
                "trigger": trigger,
                "headline": "Mem0 replaces claude-mem for HyperSwarm feed",
                "lesson_body": "Distill Mem0 session facts into mem0-session Entries.",
                "reason": "generalizable platform swap",
            }
        )

    return _fn


def _gate_rejects():
    def _fn(_prompt: str) -> str:
        return json.dumps(
            {
                "qualifies": False,
                "trigger": None,
                "headline": "",
                "lesson_body": "",
                "reason": "routine",
            }
        )

    return _fn


def test_registry_aliases():
    assert SOURCE_REGISTRY["mem0_session"] is Mem0SessionSource
    assert SOURCE_REGISTRY["mem0-session"] is Mem0SessionSource


def test_capture_lesson(tmp_path: Path):
    src = Mem0SessionSource(
        {
            "memories_fn": _memories,
            "gate_fn": _gate_qualifies("lesson"),
            "settings_path": str(tmp_path / "settings.json"),
        }
    )
    entry = src.capture(
        {
            "session_id": "sess-abc",
            "cwd": str(tmp_path / "HyperSwarmAgents"),
            "hook_event_name": "SessionEnd",
        }
    )
    assert isinstance(entry, Entry)
    assert entry.runtime == "mem0-session"
    assert entry.session_id == "mem0-sess-abc"
    assert "Mem0" in entry.summary or "mem0" in entry.summary.lower() or "lesson" in entry.body.lower()
    assert "## Trigger" in entry.body
    assert "## Lesson" in entry.body


def test_capture_leftoff_when_gate_rejects(tmp_path: Path):
    src = Mem0SessionSource(
        {
            "memories_fn": _memories,
            "gate_fn": _gate_rejects(),
            "settings_path": str(tmp_path / "settings.json"),
        }
    )
    entry = src.capture(
        {"session_id": "sess-leftoff", "cwd": "/Users/screddy/projects/Screddyice/HyperSwarmAgents"}
    )
    assert isinstance(entry, Entry)
    assert entry.runtime == "mem0-session"
    assert "leftoff" in entry.body.lower() or "Left off" in entry.summary
    assert "next_steps" in entry.body


def test_capture_none_without_session_id(tmp_path: Path):
    src = Mem0SessionSource(
        {
            "memories_fn": _memories,
            "gate_fn": _gate_qualifies(),
            "settings_path": str(tmp_path / "settings.json"),
        }
    )
    assert src.capture({"cwd": "/tmp"}) is None


def test_capture_none_without_memories(tmp_path: Path):
    src = Mem0SessionSource(
        {
            "memories_fn": lambda _sid: [],
            "gate_fn": _gate_qualifies(),
            "settings_path": str(tmp_path / "settings.json"),
        }
    )
    assert src.capture({"session_id": "empty", "cwd": "/tmp"}) is None


def test_install_idempotent(tmp_path: Path):
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"hooks": {}}))
    src = Mem0SessionSource(
        {
            "settings_path": str(settings),
            "hook_command": "hyperswarm capture --runtime mem0_session || true",
        }
    )
    src.install()
    src.install()
    data = json.loads(settings.read_text())
    cmds = [
        h["command"]
        for e in data["hooks"]["SessionEnd"]
        for h in e["hooks"]
        if h.get("type") == "command"
    ]
    assert sum(1 for c in cmds if "mem0_session" in c) == 1


def test_install_migrates_claude_mem_hook(tmp_path: Path):
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "hooks": {
                    "SessionEnd": [
                        {
                            "matcher": ".*",
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "hyperswarm capture --runtime claude_mem_session || true",
                                }
                            ],
                        }
                    ]
                }
            }
        )
    )
    src = Mem0SessionSource(
        {
            "settings_path": str(settings),
            "hook_command": "hyperswarm capture --runtime mem0_session || true",
        }
    )
    src.install()
    data = json.loads(settings.read_text())
    cmd = data["hooks"]["SessionEnd"][0]["hooks"][0]["command"]
    assert "mem0_session" in cmd
    assert "claude_mem_session" not in cmd
