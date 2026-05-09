"""Tests for OpenClawCorpusCollector — pairing turns into OpenAI fine-tune format."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperswarm.tuners.openclaw_corpus import (
    OpenClawCorpusCollector,
    TurnPair,
    _filter_pair,
    _iter_pairs_from_lines,
    pair_to_example,
)


def _write_session(tmp_path: Path, sid: str, lines: list[dict]) -> Path:
    sdir = tmp_path / "agents" / "jarvis" / "sessions"
    sdir.mkdir(parents=True, exist_ok=True)
    p = sdir / f"{sid}.jsonl"
    with open(p, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    return p


def _user(text: str) -> dict:
    return {
        "type": "message",
        "message": {"role": "user", "content": text},
        "timestamp": "2026-05-09T05:00:00Z",
    }


def _assistant(text: str) -> dict:
    return {
        "type": "message",
        "message": {"role": "assistant", "content": text},
        "timestamp": "2026-05-09T05:00:01Z",
    }


# ----------------------------------------------------------------- pairing


def test_iter_pairs_basic():
    lines = [
        json.dumps(_user("hey jarvis can you ping the team")),
        json.dumps(_assistant("Sure, drafting a quick message now")),
    ]
    pairs = list(_iter_pairs_from_lines(iter(lines)))
    assert len(pairs) == 1
    assert pairs[0].user == "hey jarvis can you ping the team"
    assert pairs[0].assistant == "Sure, drafting a quick message now"


def test_iter_pairs_drops_orphan_user_when_followed_by_user():
    """If two user messages appear back to back (no assistant in between),
    the older user message is overwritten — we only emit a pair on assistant."""
    lines = [
        json.dumps(_user("first")),
        json.dumps(_user("second")),
        json.dumps(_assistant("response to second")),
    ]
    pairs = list(_iter_pairs_from_lines(iter(lines)))
    assert len(pairs) == 1
    assert pairs[0].user == "second"


def test_iter_pairs_skips_tool_results():
    lines = [
        json.dumps(_user("run the build")),
        json.dumps({"type": "message", "message": {"role": "toolResult", "content": "..."}}),
        json.dumps(_assistant("Build is green")),
    ]
    pairs = list(_iter_pairs_from_lines(iter(lines)))
    assert len(pairs) == 1
    assert pairs[0].assistant == "Build is green"


def test_iter_pairs_skips_session_header_and_other_event_types():
    lines = [
        json.dumps({"type": "session", "id": "abc"}),
        json.dumps({"type": "model_change", "modelId": "gpt-5"}),
        json.dumps(_user("hi")),
        json.dumps(_assistant("hey there")),
    ]
    pairs = list(_iter_pairs_from_lines(iter(lines)))
    assert len(pairs) == 1


def test_iter_pairs_assistant_with_list_content():
    lines = [
        json.dumps(_user("hello")),
        json.dumps(
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Sure, "},
                        {"type": "text", "text": "happy to help."},
                    ],
                },
                "timestamp": "t",
            }
        ),
    ]
    pairs = list(_iter_pairs_from_lines(iter(lines)))
    # List content gets joined with spaces; we check fragments are present, not exact whitespace.
    assert "Sure," in pairs[0].assistant
    assert "happy to help." in pairs[0].assistant


# ----------------------------------------------------------------- filtering


def test_filter_rejects_too_short():
    p = TurnPair(user="hi", assistant="ok thanks", timestamp="t")
    assert not _filter_pair(p, min_user_chars=10, min_assistant_chars=20, max_chars=8000)


def test_filter_rejects_oversize():
    p = TurnPair(user="x" * 9000, assistant="ok this is a fine response actually", timestamp="t")
    assert not _filter_pair(p, min_user_chars=5, min_assistant_chars=5, max_chars=8000)


def test_filter_accepts_normal_size():
    p = TurnPair(
        user="hey jarvis can you pull the linear board",
        assistant="Sure, fetching the active items now from team nebula board",
        timestamp="t",
    )
    assert _filter_pair(p, min_user_chars=10, min_assistant_chars=20, max_chars=8000)


# ------------------------------------------------------------- pair_to_example


def test_pair_to_example_shape_matches_openai_fine_tune_format():
    p = TurnPair(user="what is rs21", assistant="They're an analytics consultancy.", timestamp="t")
    ex = pair_to_example(p, agent="jarvis")
    assert "messages" in ex
    msgs = ex["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert "jarvis" in msgs[0]["content"]
    assert "Shawn" in msgs[0]["content"]
    assert msgs[1]["content"] == "what is rs21"
    assert msgs[2]["content"] == "They're an analytics consultancy."


def test_pair_to_example_custom_system_prompt_overrides_default():
    p = TurnPair(user="hi" * 10, assistant="hello there friend, how can I help today", timestamp="t")
    ex = pair_to_example(p, agent="jarvis", system_prompt="You are a butler.")
    assert ex["messages"][0]["content"] == "You are a butler."


# ---------------------------------------------------------------- end-to-end


def test_collector_appends_examples_and_tracks_cursor(tmp_path: Path):
    _write_session(
        tmp_path,
        "session-1",
        [
            _user("ping the team about the deploy at 3pm"),
            _assistant("Drafting a Slack message now, will route to ops channel"),
            _user("also create a linear ticket for it"),
            _assistant("Created ticket TN-123 assigned to clawdbot for follow-up"),
        ],
    )
    c = OpenClawCorpusCollector(
        agent="jarvis",
        host="trc-server",
        agents_dir=tmp_path / "agents",
        corpus_base=tmp_path / "tune",
        state_dir=tmp_path / "state",
    )
    res = c.run()
    assert res["appended"] == 2
    assert res["total_examples"] == 2

    # Re-run: cursor advanced, no duplicate appends
    res2 = c.run()
    assert res2["appended"] == 0
    assert res2["total_examples"] == 2

    # Validate corpus file is well-formed JSONL with OpenAI shape
    corpus = (tmp_path / "tune" / "jarvis" / "corpus.jsonl").read_text().strip().splitlines()
    assert len(corpus) == 2
    for line in corpus:
        ex = json.loads(line)
        assert ex["messages"][0]["role"] == "system"
        assert ex["messages"][1]["role"] == "user"
        assert ex["messages"][2]["role"] == "assistant"


def test_collector_filters_short_pairs(tmp_path: Path):
    _write_session(
        tmp_path,
        "session-noise",
        [
            _user("hi"),  # too short, dropped
            _assistant("hello"),
            _user("here is a substantive question that is long enough to keep"),
            _assistant("here is a substantive answer that is long enough to keep"),
        ],
    )
    c = OpenClawCorpusCollector(
        agent="jarvis",
        agents_dir=tmp_path / "agents",
        corpus_base=tmp_path / "tune",
        state_dir=tmp_path / "state",
    )
    res = c.run()
    assert res["appended"] == 1


def test_collector_skips_trajectory_files(tmp_path: Path):
    sdir = tmp_path / "agents" / "jarvis" / "sessions"
    sdir.mkdir(parents=True)
    (sdir / "abc.jsonl").write_text(
        json.dumps(_user("a real prompt that is long enough"))
        + "\n"
        + json.dumps(_assistant("a real response that is long enough"))
        + "\n"
    )
    (sdir / "abc.trajectory.jsonl").write_text("garbage that should not be parsed\n")
    c = OpenClawCorpusCollector(
        agent="jarvis",
        agents_dir=tmp_path / "agents",
        corpus_base=tmp_path / "tune",
        state_dir=tmp_path / "state",
    )
    res = c.run()
    assert res["appended"] == 1


def test_collector_no_sessions_dir(tmp_path: Path):
    c = OpenClawCorpusCollector(
        agent="jarvis",
        agents_dir=tmp_path / "nonexistent",
        corpus_base=tmp_path / "tune",
        state_dir=tmp_path / "state",
    )
    assert c.run()["status"] == "no-sessions-dir"
