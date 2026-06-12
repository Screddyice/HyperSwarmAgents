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


# --- Task 6: ADVERSARIAL org-isolation gate --------------------------------
# Actively try to break the boundary. Every entry tagged company=X must be
# invisible to active org=Y!=X; shared (untagged) must be visible to all; and
# none of the dangerous near-miss cases (whitespace, substring, init-only org,
# the tool path) may leak.

_ADV_ORGS = ["TMN", "Cliqk", "TRC", "Newcalgon", "HyperXOS"]


def test_adversarial_full_org_leakage_matrix(tmp_path):
    """N-org sweep: for every active org assert it sees ONLY its own scoped
    entries plus shared, and NEVER any other org's entry."""
    p = HyperSwarmMemoryProvider(root=str(tmp_path))
    (tmp_path / "entries").mkdir()
    for o in _ADV_ORGS:
        _seed(p._store, f"{o} secret keyword payload", o)
    _seed(p._store, "shared keyword payload", None)

    for active in _ADV_ORGS:
        out = p.prefetch("keyword", session_id="s", org=active)
        assert f"{active} secret" in out, f"own entry missing for {active}"
        assert "shared keyword" in out, f"shared invisible to {active}"
        for other in _ADV_ORGS:
            if other == active:
                continue
            assert f"{other} secret" not in out, (
                f"LEAK: {other} entry surfaced under active org {active}"
            )


def test_adversarial_active_org_from_initialize_only(tmp_path):
    """When the active org is set via initialize() (no per-call org=), recall
    must still isolate — a Cliqk entry must not leak into a TMN session."""
    p = HyperSwarmMemoryProvider(root=str(tmp_path))
    (tmp_path / "entries").mkdir()
    p.initialize("s", org="TMN")
    _seed(p._store, "TMN alpha keyword", "TMN")
    _seed(p._store, "Cliqk beta keyword", "Cliqk")
    out = p.prefetch("keyword", session_id="s")  # no org kwarg -> self._org
    assert "alpha" in out
    assert "beta" not in out


def test_adversarial_whitespace_padded_scope_no_leak(tmp_path):
    """A whitespace-padded scope ("  Cliqk  ") must normalize and NOT leak into
    a different active org."""
    p = HyperSwarmMemoryProvider(root=str(tmp_path))
    (tmp_path / "entries").mkdir()
    _seed(p._store, "padded cliqk keyword", "  Cliqk  ")
    out = p.prefetch("keyword", session_id="s", org="TMN")
    assert "cliqk" not in out.lower()


def test_adversarial_substring_org_name_no_leak(tmp_path):
    """Org matching is exact equality, not prefix — a 'TMNX' entry must not
    leak into a 'TMN' session."""
    p = HyperSwarmMemoryProvider(root=str(tmp_path))
    (tmp_path / "entries").mkdir()
    _seed(p._store, "TMNX other keyword", "TMNX")
    _seed(p._store, "TMN own keyword", "TMN")
    out = p.prefetch("keyword", session_id="s", org="TMN")
    assert "own keyword" in out
    assert "TMNX" not in out


def test_adversarial_tool_path_matrix_isolation(tmp_path):
    """The explicit hyperswarm_search tool must enforce the same boundary as
    prefetch across the full org matrix."""
    p = HyperSwarmMemoryProvider(root=str(tmp_path))
    (tmp_path / "entries").mkdir()
    for o in _ADV_ORGS:
        _seed(p._store, f"{o} secret keyword payload", o)
    _seed(p._store, "shared keyword payload", None)
    import json

    for active in _ADV_ORGS:
        p.initialize("s", org=active)
        res = json.loads(p.handle_tool_call("hyperswarm_search", {"query": "keyword"}))
        joined = " ".join(res["results"])
        assert f"{active} secret" in joined
        assert "shared keyword" in joined
        for other in _ADV_ORGS:
            if other == active:
                continue
            assert f"{other} secret" not in joined, (
                f"LEAK via tool: {other} surfaced under active org {active}"
            )


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


def test_register_module_has_discovery_markers():
    """Hermes' discover_memory_providers() treats a dir as a provider iff its
    source references register_memory_provider / MemoryProvider."""
    import inspect

    import hyperswarm.integrations.hermes as pkg

    src = inspect.getsource(pkg)
    assert "register_memory_provider" in src
    assert "MemoryProvider" in src


def test_register_accepts_provider_kwarg_path():
    """Some Hermes builds pass the already-built provider in via kwargs;
    register must not double-instantiate when one is supplied."""
    from hyperswarm.integrations.hermes import register

    captured = {}

    class _Ctx:
        def register_memory_provider(self, provider):
            captured["provider"] = provider

    sentinel = HyperSwarmMemoryProvider()
    register(_Ctx(), provider=sentinel)
    assert captured["provider"] is sentinel


# --- HYPERSWARM_MEMORY_DISABLE guard ----------------------------------------
# Headless one-shot callers (e.g. the meeting-prep runner's `hermes -z`
# synthesis) set HYPERSWARM_MEMORY_DISABLE=1 so their sessions neither inject
# memory context nor write "session left-off" noise into the working store.

def test_register_skips_when_disable_env_set(monkeypatch):
    from hyperswarm.integrations.hermes import register

    captured = {}

    class _Ctx:
        def register_memory_provider(self, provider):
            captured["provider"] = provider

    monkeypatch.setenv("HYPERSWARM_MEMORY_DISABLE", "1")
    register(_Ctx())
    assert "provider" not in captured


def test_register_skips_for_true_value(monkeypatch):
    from hyperswarm.integrations.hermes import register

    captured = {}

    class _Ctx:
        def register_memory_provider(self, provider):
            captured["provider"] = provider

    monkeypatch.setenv("HYPERSWARM_MEMORY_DISABLE", "true")
    register(_Ctx())
    assert "provider" not in captured


def test_register_runs_for_falsey_values(monkeypatch):
    from hyperswarm.integrations.hermes import register

    for val in ("", "0", "false", "no"):
        captured = {}

        class _Ctx:
            def register_memory_provider(self, provider):
                captured["provider"] = provider

        monkeypatch.setenv("HYPERSWARM_MEMORY_DISABLE", val)
        register(_Ctx())
        assert isinstance(captured.get("provider"), HyperSwarmMemoryProvider), val
