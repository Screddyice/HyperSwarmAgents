"""Verify Entry serialises to markdown and reads back losslessly."""
from __future__ import annotations

import datetime as _dt

from hyperswarm.core.entry import Entry


def test_roundtrip():
    ts = _dt.datetime(2026, 5, 4, 22, 14, 0, tzinfo=_dt.timezone.utc)
    e = Entry(
        runtime="claude-code",
        cwd="/Users/me/projects/teamnebula.ai/api",
        summary="Wired dual-write into watchdog",
        body="# Decisions\n\n- chose path X over Y because Z\n",
        session_id="abc123",
        scope="NEB",
        timestamp=ts,
    )
    md = e.to_markdown()
    assert md.startswith("---\n")
    assert "scope: NEB" in md
    assert "summary: Wired dual-write into watchdog" in md

    e2 = Entry.from_markdown(md)
    assert e2.runtime == e.runtime
    assert e2.cwd == e.cwd
    assert e2.summary == e.summary
    assert e2.session_id == e.session_id
    assert e2.scope == e.scope
    assert e2.timestamp == e.timestamp
    assert "chose path X over Y" in e2.body


def test_summary_is_collapsed_to_one_line():
    e = Entry(
        runtime="r", cwd="/", summary="line1\nline2\rline3",
        body="body",
    )
    md = e.to_markdown()
    summary_line = next(ln for ln in md.splitlines() if ln.startswith("summary: "))
    assert "\n" not in summary_line.removeprefix("summary: ")
    assert "line1 line2 line3" == summary_line.removeprefix("summary: ")


def test_from_markdown_rejects_missing_frontmatter():
    import pytest
    with pytest.raises(ValueError):
        Entry.from_markdown("just a body, no frontmatter")
