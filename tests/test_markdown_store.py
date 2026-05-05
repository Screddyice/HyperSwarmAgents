"""Verify MarkdownStore: write/read roundtrip and list_since filtering."""
from __future__ import annotations

import datetime as _dt

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
