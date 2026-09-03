"""Verify MarkdownStore: write/read roundtrip and list_since filtering."""
from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path

from hyperswarm.core.entry import Entry
from hyperswarm.stores.markdown import MarkdownStore


def _entry(ts: _dt.datetime, runtime: str = "test") -> Entry:
    return Entry(
        runtime=runtime,
        cwd="/cwd",
        summary=f"entry at {ts.isoformat()}",
        body="body",
        timestamp=ts,
    )


def test_write_creates_dated_path(tmp_path):
    store = MarkdownStore({"path": str(tmp_path)})
    ts = _dt.datetime(2026, 5, 4, 12, 0, 0, tzinfo=_dt.timezone.utc)
    sid = store.write(_entry(ts))
    assert "/2026/05/04/" in sid
    assert sid.endswith(".md")


def test_read_returns_equivalent_entry(tmp_path):
    store = MarkdownStore({"path": str(tmp_path)})
    e = _entry(_dt.datetime(2026, 5, 4, 12, 0, 0, tzinfo=_dt.timezone.utc))
    sid = store.write(e)
    e2 = store.read(sid)
    assert e2.runtime == e.runtime
    assert e2.summary == e.summary
    assert e2.timestamp == e.timestamp


def test_list_since_filters_by_timestamp(tmp_path):
    store = MarkdownStore({"path": str(tmp_path)})
    older = _dt.datetime(2026, 5, 1, 10, 0, 0, tzinfo=_dt.timezone.utc)
    newer = _dt.datetime(2026, 5, 4, 22, 0, 0, tzinfo=_dt.timezone.utc)
    store.write(_entry(older))
    store.write(_entry(newer))

    cutoff = _dt.datetime(2026, 5, 3, 0, 0, 0, tzinfo=_dt.timezone.utc)
    results = list(store.list_since(cutoff))
    assert len(results) == 1
    assert results[0].timestamp == newer


def test_list_since_returns_empty_when_root_missing(tmp_path):
    store = MarkdownStore({"path": str(tmp_path / "does-not-exist")})
    results = list(store.list_since(_dt.datetime.min.replace(tzinfo=_dt.timezone.utc)))
    assert results == []


def test_default_root_is_projects_on_mac(monkeypatch):
    """With no explicit path, the Mac stores under the projects-root convention."""
    monkeypatch.setenv("HYPERSWARM_HOST_IDENTITY", "shawn-mac")
    store = MarkdownStore({})
    assert store.root == Path(os.path.expanduser("~/projects/HyperSwarm"))


def test_default_root_is_home_hyperswarm_on_servers(monkeypatch):
    """Every non-Mac host keeps the historical ~/HyperSwarm default."""
    monkeypatch.setenv("HYPERSWARM_HOST_IDENTITY", "neb-server")
    store = MarkdownStore({})
    assert store.root == Path(os.path.expanduser("~/HyperSwarm"))


def test_explicit_path_overrides_host_default(monkeypatch, tmp_path):
    """An explicit `path` always wins, regardless of host identity."""
    monkeypatch.setenv("HYPERSWARM_HOST_IDENTITY", "shawn-mac")
    store = MarkdownStore({"path": str(tmp_path)})
    assert store.root == tmp_path


def test_write_sets_storage_id_on_entry(tmp_path):
    store = MarkdownStore({"path": str(tmp_path)})
    e = _entry(_dt.datetime(2026, 6, 12, 12, 0, 0, tzinfo=_dt.timezone.utc))
    sid = store.write(e)
    assert e.storage_id == sid


def test_read_populates_storage_id(tmp_path):
    store = MarkdownStore({"path": str(tmp_path)})
    sid = store.write(_entry(_dt.datetime(2026, 6, 12, 12, 0, 0, tzinfo=_dt.timezone.utc)))
    e = store.read(sid)
    assert e.storage_id == sid


def test_list_since_populates_storage_id(tmp_path):
    """Consumers (corpus ingestion dedup + cursor tiebreak) need a stable
    per-entry identity; without it source_entry_path is NULL downstream."""
    store = MarkdownStore({"path": str(tmp_path)})
    ts = _dt.datetime(2026, 6, 12, 12, 0, 0, tzinfo=_dt.timezone.utc)
    sid = store.write(_entry(ts))
    results = list(store.list_since(ts - _dt.timedelta(days=1)))
    assert len(results) == 1
    assert results[0].storage_id == sid


def test_storage_id_not_serialized_into_markdown(tmp_path):
    """storage_id is storage metadata, not entry content — it must not leak
    into the frontmatter (it would go stale the moment a file is synced)."""
    store = MarkdownStore({"path": str(tmp_path)})
    sid = store.write(_entry(_dt.datetime(2026, 6, 12, 12, 0, 0, tzinfo=_dt.timezone.utc)))
    assert "storage_id" not in Path(sid).read_text()
