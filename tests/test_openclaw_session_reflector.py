"""Tests for the openclaw session reflector.

Covers parsing one jsonl turn, full reflector run with mocked LLM, cursor
idempotency, and frontmatter provenance injection.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from hyperswarm.reflectors.openclaw_session import (
    OpenClawSessionReflector,
    _inject_provenance,
    _semantic_hash,
    extract_turn,
    read_new_turns,
    split_memory_blocks,
    write_memory_block,
)


# ----------------------------------------------------------------- extract_turn


def test_extract_turn_user():
    line = json.dumps(
        {
            "type": "message",
            "id": "abc",
            "timestamp": "2026-05-09T04:49:00Z",
            "message": {"role": "user", "content": "hey jarvis can you ping the team"},
        }
    )
    t = extract_turn(line)
    assert t is not None
    assert t.role == "user"
    assert "ping the team" in t.text
    assert t.timestamp == "2026-05-09T04:49:00Z"


def test_extract_turn_assistant_list_content():
    line = json.dumps(
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Sure, "},
                    {"type": "text", "text": "sending now."},
                ],
            },
            "timestamp": "2026-05-09T04:49:01Z",
        }
    )
    t = extract_turn(line)
    assert t is not None
    assert t.role == "assistant"
    assert "Sure" in t.text and "sending now" in t.text


def test_extract_turn_skips_non_message():
    for typ in ("session", "model_change", "thinking_level_change", "toolCall"):
        line = json.dumps({"type": typ, "id": "x"})
        assert extract_turn(line) is None


def test_extract_turn_skips_tool_role():
    line = json.dumps(
        {"type": "message", "message": {"role": "toolResult", "content": "{}"}}
    )
    assert extract_turn(line) is None


def test_extract_turn_truncates_long_text():
    line = json.dumps(
        {
            "type": "message",
            "message": {"role": "user", "content": "x" * 5000},
            "timestamp": "t",
        }
    )
    t = extract_turn(line, max_chars=100)
    assert t is not None
    assert len(t.text) <= 101  # 100 + ellipsis
    assert t.text.endswith("…")


def test_extract_turn_handles_malformed_json():
    assert extract_turn("not-json{{") is None


# --------------------------------------------------------------- read_new_turns


def _write_session(tmp_path: Path, sid: str, lines: list[dict]) -> Path:
    sdir = tmp_path / "agents" / "jarvis" / "sessions"
    sdir.mkdir(parents=True, exist_ok=True)
    p = sdir / f"{sid}.jsonl"
    with open(p, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    return p


def test_read_new_turns_picks_up_after_cursor(tmp_path: Path):
    p = _write_session(
        tmp_path,
        "s1",
        [
            {"type": "message", "message": {"role": "user", "content": "first"}, "timestamp": "t1"},
            {"type": "message", "message": {"role": "assistant", "content": "second"}, "timestamp": "t2"},
        ],
    )
    turns, offset = read_new_turns(p, cursor_offset=0)
    assert len(turns) == 2
    assert offset > 0
    # Reading again at the new offset should yield zero new turns.
    turns2, offset2 = read_new_turns(p, cursor_offset=offset)
    assert turns2 == []
    assert offset2 == offset


def test_read_new_turns_caps_at_max_turns(tmp_path: Path):
    lines = [
        {"type": "message", "message": {"role": "user", "content": f"msg {i}"}, "timestamp": f"t{i}"}
        for i in range(100)
    ]
    p = _write_session(tmp_path, "s2", lines)
    turns, _ = read_new_turns(p, max_turns=10)
    # Should be the LAST 10
    assert len(turns) == 10
    assert turns[-1].text == "msg 99"
    assert turns[0].text == "msg 90"


# ------------------------------------------------------------- block splitting


def test_split_memory_blocks_two_blocks():
    out = """---
name: Shawn prefers Markdown-only emails
description: never use HTML in outbound mail
type: feedback
---
**Why:** he formats by hand and HTML breaks the paste flow.

**How to apply:** plain text emails always.
---
name: TRC server lives on its own AWS account
description: account 067960554345 not 429835537523
type: project
---
TRC openclaw is on a separate AWS account from NEB+Cliqk. Use aws --profile trc.
"""
    blocks = split_memory_blocks(out)
    assert len(blocks) == 2
    assert "Markdown-only" in blocks[0]
    assert "067960554345" in blocks[1]
    assert blocks[0].startswith("---\n")
    assert blocks[0].rstrip().endswith("plain text emails always.")


def test_split_memory_blocks_empty_string():
    assert split_memory_blocks("") == []
    assert split_memory_blocks("   ") == []


def test_split_memory_blocks_malformed_no_frontmatter():
    # No leading `---` → reject everything (safer than emitting garbage)
    assert split_memory_blocks("just some text without frontmatter") == []


# --------------------------------------------------------- semantic dedup


def test_semantic_hash_stable_for_same_name_description():
    a = "---\nname: Shawn prefers plain text\ndescription: never HTML\ntype: feedback\n---\nbody A\n"
    b = "---\nname: Shawn prefers plain text\ndescription: never HTML\ntype: feedback\n---\ncompletely different body B\n"
    assert _semantic_hash(a) == _semantic_hash(b), "different bodies but same name+desc → same hash"


def test_semantic_hash_differs_when_name_or_description_differs():
    a = "---\nname: A thing\ndescription: hook\ntype: user\n---\nbody\n"
    b = "---\nname: A different thing\ndescription: hook\ntype: user\n---\nbody\n"
    c = "---\nname: A thing\ndescription: different hook\ntype: user\n---\nbody\n"
    assert _semantic_hash(a) != _semantic_hash(b)
    assert _semantic_hash(a) != _semantic_hash(c)


def test_semantic_hash_case_insensitive():
    a = "---\nname: meeting prep workflow\ndescription: how to run it\ntype: user\n---\nbody\n"
    b = "---\nname: Meeting Prep Workflow\ndescription: How to run it\ntype: user\n---\nbody\n"
    assert _semantic_hash(a) == _semantic_hash(b)


def test_write_memory_block_skips_duplicate_by_name_description(tmp_path):
    block = "---\nname: meeting prep workflow\ndescription: how to run it\ntype: user\n---\nworkflow body\n"
    p1 = write_memory_block(
        block=block,
        agent="clawdbot",
        host="neb-server",
        session_id="aaa11111-1111-1111",
        timestamp="2026-05-09T00:00:00Z",
        output_dir=tmp_path,
    )
    assert p1 is not None and p1.exists()

    # Second write with same name+description from a DIFFERENT session: should skip
    block2 = "---\nname: meeting prep workflow\ndescription: how to run it\ntype: user\n---\nslightly different body\n"
    p2 = write_memory_block(
        block=block2,
        agent="clawdbot",
        host="neb-server",
        session_id="bbb22222-2222-2222",
        timestamp="2026-05-09T01:00:00Z",
        output_dir=tmp_path,
    )
    assert p2 is None

    # Only one .md file in dir
    md_files = list(tmp_path.glob("*.md"))
    assert len(md_files) == 1


def test_write_memory_block_writes_when_name_or_description_differs(tmp_path):
    write_memory_block(
        block="---\nname: A\ndescription: hook A\ntype: user\n---\nbody\n",
        agent="x", host="h", session_id="s1", timestamp="t",
        output_dir=tmp_path,
    )
    write_memory_block(
        block="---\nname: B\ndescription: hook A\ntype: user\n---\nbody\n",
        agent="x", host="h", session_id="s2", timestamp="t",
        output_dir=tmp_path,
    )
    write_memory_block(
        block="---\nname: A\ndescription: hook B\ntype: user\n---\nbody\n",
        agent="x", host="h", session_id="s3", timestamp="t",
        output_dir=tmp_path,
    )
    assert len(list(tmp_path.glob("*.md"))) == 3


# ------------------------------------------------------- provenance injection


def test_inject_provenance_adds_missing_fields():
    block = """---
name: Test memory
description: hook
type: user
---
body content here.
"""
    out = _inject_provenance(
        block,
        agent="jarvis",
        host="trc-server",
        session_id="abc-123",
        timestamp="2026-05-09T05:00:00Z",
    )
    assert "originAgent: jarvis" in out
    assert "originHost: trc-server" in out
    assert "originSession: abc-123" in out
    assert "originTimestamp: 2026-05-09T05:00:00Z" in out
    # Existing fields preserved
    assert "name: Test memory" in out
    assert "body content here." in out


def test_inject_provenance_does_not_duplicate():
    block = """---
name: Existing
description: hook
type: user
originAgent: existing-agent
---
body.
"""
    out = _inject_provenance(block, agent="jarvis", host="trc-server", session_id="x", timestamp="t")
    # originAgent should remain the existing-agent value, not be overwritten
    assert "originAgent: existing-agent" in out
    # but originHost / originSession / originTimestamp are added
    assert "originHost: trc-server" in out
    assert out.count("originAgent:") == 1


# -------------------------------------------------------------- end-to-end run


def test_reflector_end_to_end_with_mock_llm(tmp_path: Path):
    """Full pass: read session → mock LLM → write blocks → save state.
    Re-run is idempotent (same hashes → no duplicate files, cursor stable)."""
    _write_session(
        tmp_path,
        "session-aaa",
        [
            {
                "type": "message",
                "message": {"role": "user", "content": "always use plain text in emails"},
                "timestamp": "2026-05-09T04:49:00Z",
            },
            {
                "type": "message",
                "message": {"role": "assistant", "content": "got it, plain text only."},
                "timestamp": "2026-05-09T04:49:02Z",
            },
        ],
    )
    output_base = tmp_path / "memory" / "server-learned"
    state_dir = tmp_path / "state"

    mock_llm_response = """---
name: Shawn prefers plain text emails
description: never HTML in outbound mail
type: feedback
---
**Why:** explicit instruction during this session.

**How to apply:** always plain text for any email Shawn drafts.
"""

    def mock_llm(messages):
        # Verify the prompt is reasonable
        assert messages[0]["role"] == "system"
        assert "high-signal" in messages[0]["content"].lower() or "high signal" in messages[0]["content"].lower()
        assert "session-aaa" in messages[1]["content"]
        return mock_llm_response

    r = OpenClawSessionReflector(
        agent="jarvis",
        host="trc-server",
        agents_dir=tmp_path / "agents",
        output_base=output_base,
        state_dir=state_dir,
        llm_call=mock_llm,
    )
    result = r.run()

    assert result["sessions_processed"] == 1
    assert result["written"] == 1
    written_path = Path(result["files"][0])
    assert written_path.exists()
    text = written_path.read_text()
    assert "originAgent: jarvis" in text
    assert "originHost: trc-server" in text
    assert "originSession: session-aaa" in text
    assert "Shawn prefers plain text emails" in text

    # Re-run: cursor advanced, no new turns, no new writes
    result2 = r.run()
    assert result2["sessions_processed"] == 0
    assert result2["written"] == 0


def test_reflector_skips_when_llm_returns_nothing(tmp_path: Path):
    _write_session(
        tmp_path,
        "session-empty",
        [{"type": "message", "message": {"role": "user", "content": "what time is it"}, "timestamp": "t"}],
    )
    r = OpenClawSessionReflector(
        agent="jarvis",
        host="x",
        agents_dir=tmp_path / "agents",
        output_base=tmp_path / "out",
        state_dir=tmp_path / "state",
        llm_call=lambda _msgs: "",
    )
    res = r.run()
    assert res["sessions_processed"] == 1
    assert res["written"] == 0


def test_reflector_no_sessions_dir(tmp_path: Path):
    r = OpenClawSessionReflector(
        agent="jarvis",
        agents_dir=tmp_path / "nonexistent",
        output_base=tmp_path / "out",
        state_dir=tmp_path / "state",
        llm_call=lambda _msgs: "",
    )
    assert r.run()["status"] == "no-sessions-dir"


def test_reflector_skips_trajectory_files(tmp_path: Path):
    """Sessions dir contains both `*.jsonl` (events) and `*.trajectory.jsonl` (different schema).
    Reflector must read only the former."""
    sdir = tmp_path / "agents" / "jarvis" / "sessions"
    sdir.mkdir(parents=True)
    # The event file (should be processed)
    (sdir / "abc.jsonl").write_text(
        json.dumps({"type": "message", "message": {"role": "user", "content": "hi"}, "timestamp": "t"})
        + "\n"
    )
    # A trajectory file (should be skipped — different schema)
    (sdir / "abc.trajectory.jsonl").write_text("garbage that would crash the parser if read\n")

    calls: list[str] = []

    def mock_llm(msgs):
        calls.append(msgs[1]["content"])
        return ""  # no memory

    r = OpenClawSessionReflector(
        agent="jarvis",
        agents_dir=tmp_path / "agents",
        output_base=tmp_path / "out",
        state_dir=tmp_path / "state",
        llm_call=mock_llm,
    )
    r.run()
    # Exactly one LLM call (for abc.jsonl), trajectory file ignored
    assert len(calls) == 1
    assert "abc" in calls[0]
