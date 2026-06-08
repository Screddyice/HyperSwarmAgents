"""Tests for the Hermes HyperSwarm memory provider.

Org isolation is the security boundary: an entry scoped to one company must
NEVER surface in recall when a different company is the active org. Shared
(unscoped) entries are visible to all orgs.

The org/company tag is carried by ``Entry.scope`` (a string), NOT a
``frontmatter`` dict — see hyperswarm/core/{entry,scope}.py.
"""
from __future__ import annotations

import datetime as _dt

from hyperswarm.core.entry import Entry
from hyperswarm.integrations.hermes.provider import HyperSwarmMemoryProvider


def _epoch() -> _dt.datetime:
    return _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc)


def _seed(store, body: str, company: str | None) -> None:
    """Write an entry whose org tag is carried by ``Entry.scope``."""
    e = Entry(
        runtime="test",
        cwd="/cwd",
        summary=body[:40],
        body=body,
        scope=company or "",
    )
    store.write(e)


# --- Task 1: scaffold ------------------------------------------------------

def test_provider_name_and_availability(tmp_path):
    p = HyperSwarmMemoryProvider(root=str(tmp_path))
    assert p.name == "hyperswarm"
    (tmp_path / "entries").mkdir()
    assert p.is_available() is True


# --- Task 2: recall with org isolation -------------------------------------

def test_prefetch_org_isolation(tmp_path):
    p = HyperSwarmMemoryProvider(root=str(tmp_path))
    (tmp_path / "entries").mkdir()
    _seed(p._store, "TMN lead pipeline note alpha", "TMN")
    _seed(p._store, "Cliqk secret campaign beta pipeline", "Cliqk")
    _seed(p._store, "Shawn prefers plain-text email", None)  # shared

    out = p.prefetch("pipeline", session_id="s", org="TMN")
    assert "alpha" in out                 # TMN entry recalled
    assert "beta" not in out              # Cliqk entry NEVER leaks into TMN recall


def test_prefetch_shared_entries_visible_to_all_orgs(tmp_path):
    p = HyperSwarmMemoryProvider(root=str(tmp_path))
    (tmp_path / "entries").mkdir()
    _seed(p._store, "Shawn prefers plain-text email always", None)  # shared
    out = p.prefetch("email", session_id="s", org="TMN")
    assert "plain-text" in out


def test_prefetch_cross_org_matrix(tmp_path):
    """No entry tagged company=X is ever returned when active org=Y != X."""
    p = HyperSwarmMemoryProvider(root=str(tmp_path))
    (tmp_path / "entries").mkdir()
    _seed(p._store, "TMN keyword token", "TMN")
    _seed(p._store, "Cliqk keyword token", "Cliqk")
    _seed(p._store, "TRC keyword token", "TRC")
    _seed(p._store, "shared keyword token", None)

    for active, allowed, forbidden in [
        ("TMN", "TMN", ("Cliqk", "TRC")),
        ("Cliqk", "Cliqk", ("TMN", "TRC")),
        ("TRC", "TRC", ("TMN", "Cliqk")),
    ]:
        out = p.prefetch("keyword", session_id="s", org=active)
        assert f"{allowed} keyword" in out
        assert "shared keyword" in out  # shared visible to all
        for f in forbidden:
            assert f"{f} keyword" not in out


# --- Task 3: write path (capture-scope safe) -------------------------------

def test_write_is_session_scoped_not_per_turn(tmp_path):
    p = HyperSwarmMemoryProvider(root=str(tmp_path))
    (tmp_path / "entries").mkdir()
    p.initialize("s1", org="TMN")
    p.sync_turn("hi", "hello")  # must NOT write a per-turn entry
    assert sum(1 for _ in p._store.list_since(_epoch())) == 0

    p.on_session_end([
        {"role": "user", "content": "ship the deck"},
        {"role": "assistant", "content": "done, deck shipped"},
    ])
    entries = list(p._store.list_since(_epoch()))
    assert len(entries) == 1                      # exactly one left-off entry
    assert p._entry_company(entries[0]) == "TMN"  # scope-tagged with active org


def test_session_end_no_buffer_writes_nothing(tmp_path):
    p = HyperSwarmMemoryProvider(root=str(tmp_path))
    (tmp_path / "entries").mkdir()
    p.initialize("s1", org="TMN")
    p.on_session_end([])
    assert sum(1 for _ in p._store.list_since(_epoch())) == 0


# --- Task 4: explicit recall tool (get_tool_schemas / handle_tool_call) -----

def test_search_tool_schema_exposed(tmp_path):
    p = HyperSwarmMemoryProvider(root=str(tmp_path))
    (tmp_path / "entries").mkdir()
    schemas = p.get_tool_schemas()
    assert any(s["name"] == "hyperswarm_search" for s in schemas)
    schema = next(s for s in schemas if s["name"] == "hyperswarm_search")
    # The description MUST route deep/personal/health/broad queries to the
    # corpus-mcp brain, keeping HyperSwarm for working memory.
    desc = schema["description"].lower()
    assert "corpus" in desc
    assert "working memory" in desc
    for kw in ("deep", "personal", "health", "broad"):
        assert kw in desc, f"description must route {kw!r} queries to corpus"
    assert schema["parameters"]["required"] == ["query"]


def test_search_tool_returns_org_filtered_hits(tmp_path):
    p = HyperSwarmMemoryProvider(root=str(tmp_path))
    (tmp_path / "entries").mkdir()
    p.initialize("s", org="TMN")
    _seed(p._store, "alpha pipeline note", "TMN")
    import json

    res = json.loads(p.handle_tool_call("hyperswarm_search", {"query": "pipeline"}))
    assert any("alpha" in r for r in res["results"])


def test_search_tool_org_isolation(tmp_path):
    """The tool path must enforce the same org boundary as prefetch()."""
    p = HyperSwarmMemoryProvider(root=str(tmp_path))
    (tmp_path / "entries").mkdir()
    p.initialize("s", org="TMN")
    _seed(p._store, "TMN alpha keyword token", "TMN")
    _seed(p._store, "Cliqk beta keyword token", "Cliqk")
    import json

    res = json.loads(p.handle_tool_call("hyperswarm_search", {"query": "keyword"}))
    joined = " ".join(res["results"])
    assert "alpha" in joined
    assert "beta" not in joined  # Cliqk entry NEVER leaks via the tool


def test_search_tool_no_hits_empty_results(tmp_path):
    p = HyperSwarmMemoryProvider(root=str(tmp_path))
    (tmp_path / "entries").mkdir()
    p.initialize("s", org="TMN")
    import json

    res = json.loads(p.handle_tool_call("hyperswarm_search", {"query": "nothingmatches"}))
    assert res["results"] == []


def test_unknown_tool_raises(tmp_path):
    p = HyperSwarmMemoryProvider(root=str(tmp_path))
    (tmp_path / "entries").mkdir()
    import pytest

    with pytest.raises(NotImplementedError):
        p.handle_tool_call("not_a_tool", {})


# --- Task 5: registration entrypoint ---------------------------------------

def test_register_entrypoint_calls_register_memory_provider():
    from hyperswarm.integrations.hermes import register

    captured = {}

    class _Ctx:
        def register_memory_provider(self, provider):
            captured["provider"] = provider

    register(_Ctx())
    assert isinstance(captured.get("provider"), HyperSwarmMemoryProvider)
    assert captured["provider"].name == "hyperswarm"
