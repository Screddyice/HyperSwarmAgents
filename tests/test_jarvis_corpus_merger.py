"""Tests for JarvisCorpusMerger — cross-node Jarvis corpus merge.

The merger pulls session JSONLs from multiple hosts (rsync; injectable in
tests), then walks them with host-namespaced cursors and writes one merged
corpus.jsonl. These tests verify:

- rsync command construction and per-host failure isolation
- Local sources are read directly without rsync
- Cursor namespacing keeps two hosts with same session UUID independent
- Re-runs are idempotent (no duplicate examples)
- Pair filtering matches the per-host collector's filter
- Unique-host invariant catches bad config
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperswarm.tuners.jarvis_merge import CorpusSource, JarvisCorpusMerger


# ── helpers ────────────────────────────────────────────────────────────


def _write_session(dir: Path, sid: str, pairs: list[tuple[str, str]]) -> Path:
    """Write a fake openclaw/Claude-Code session jsonl to `dir/<sid>.jsonl`.
    Each pair becomes one user message followed by one assistant message."""
    dir.mkdir(parents=True, exist_ok=True)
    p = dir / f"{sid}.jsonl"
    lines = []
    for i, (u, a) in enumerate(pairs):
        lines.append(json.dumps({"type": "message", "message": {"role": "user", "content": u}, "timestamp": f"t{i}u"}))
        lines.append(json.dumps({"type": "message", "message": {"role": "assistant", "content": a}, "timestamp": f"t{i}a"}))
    p.write_text("\n".join(lines) + "\n")
    return p


def _make_runner(*, fail_hosts: set[str] | None = None):
    """Returns a fake rsync runner that copies src dir contents (a bare path
    on disk that we pre-seeded) into dst. Failed hosts return rc=1."""
    fail_hosts = fail_hosts or set()
    seen: list[dict] = []

    def runner(*, src: str, dst: Path):
        # src is "<alias>:<path>/" — for the fake we ignore the alias and
        # just resolve <path> on the local filesystem (the test seeds it
        # there).
        seen.append({"src": src, "dst": str(dst)})
        # Extract host from src ("alias:path/")
        alias = src.split(":", 1)[0]
        if alias in fail_hosts:
            return 1, "permission denied"
        # Extract real path: strip "alias:" prefix
        local_src = src.split(":", 1)[1].rstrip("/")
        src_path = Path(local_src)
        if not src_path.exists():
            return 1, f"source missing {src_path}"
        dst.mkdir(parents=True, exist_ok=True)
        for f in src_path.glob("*.jsonl"):
            (dst / f.name).write_bytes(f.read_bytes())
        return 0, ""

    runner.seen = seen  # type: ignore[attr-defined]
    return runner


# ── unique-host guard ─────────────────────────────────────────────────


def test_duplicate_host_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="duplicate host"):
        JarvisCorpusMerger(
            sources=[
                CorpusSource(host="x", remote_path="/a"),
                CorpusSource(host="x", remote_path="/b"),
            ],
            stage_base=tmp_path / "stage",
            corpus_base=tmp_path / "corpus",
            state_dir=tmp_path / "state",
        )


# ── pull_remotes ──────────────────────────────────────────────────────


def test_pull_remotes_calls_rsync_per_remote_skips_local(tmp_path: Path):
    # Seed a fake "remote" path on disk
    fake_neb = tmp_path / "fake_neb_sessions"
    fake_neb.mkdir()
    (fake_neb / "abc.jsonl").write_text("")

    runner = _make_runner()
    merger = JarvisCorpusMerger(
        sources=[
            CorpusSource(host="mac", remote_path=str(tmp_path / "mac"), ssh_alias=None),
            CorpusSource(host="neb-server", remote_path=str(fake_neb), ssh_alias="neb-server"),
        ],
        stage_base=tmp_path / "stage",
        corpus_base=tmp_path / "corpus",
        state_dir=tmp_path / "state",
        rsync_runner=runner,
    )
    res = merger.pull_remotes()

    assert res["mac"]["status"] == "local"
    assert res["neb-server"]["status"] == "pulled"
    assert len(runner.seen) == 1
    assert runner.seen[0]["src"].startswith("neb-server:")
    assert (tmp_path / "stage" / "neb-server" / "abc.jsonl").exists()


def test_pull_remotes_isolates_per_host_failure(tmp_path: Path):
    fake_neb = tmp_path / "fake_neb"
    fake_neb.mkdir()
    (fake_neb / "n1.jsonl").write_text("")
    fake_trc = tmp_path / "fake_trc"
    fake_trc.mkdir()
    (fake_trc / "t1.jsonl").write_text("")

    runner = _make_runner(fail_hosts={"trc-server"})
    merger = JarvisCorpusMerger(
        sources=[
            CorpusSource(host="neb-server", remote_path=str(fake_neb), ssh_alias="neb-server"),
            CorpusSource(host="trc-server", remote_path=str(fake_trc), ssh_alias="trc-server"),
        ],
        stage_base=tmp_path / "stage",
        corpus_base=tmp_path / "corpus",
        state_dir=tmp_path / "state",
        rsync_runner=runner,
    )
    res = merger.pull_remotes()
    assert res["neb-server"]["status"] == "pulled"
    assert res["trc-server"]["status"] == "failed"
    assert "permission denied" in res["trc-server"]["reason"]


def test_pull_remotes_runner_crash_does_not_abort_others(tmp_path: Path):
    fake_neb = tmp_path / "fake_neb"
    fake_neb.mkdir()
    (fake_neb / "n1.jsonl").write_text("")

    def bad_runner(*, src, dst):
        raise RuntimeError("ssh exploded")

    merger = JarvisCorpusMerger(
        sources=[
            CorpusSource(host="neb-server", remote_path=str(fake_neb), ssh_alias="neb"),
        ],
        stage_base=tmp_path / "stage",
        corpus_base=tmp_path / "corpus",
        state_dir=tmp_path / "state",
        rsync_runner=bad_runner,
    )
    res = merger.pull_remotes()
    assert res["neb-server"]["status"] == "failed"
    assert "ssh exploded" in res["neb-server"]["reason"]


# ── collect ──────────────────────────────────────────────────────────


def test_collect_local_source_reads_remote_path_directly(tmp_path: Path):
    mac_dir = tmp_path / "mac"
    _write_session(mac_dir, "session-a", [
        ("hello there please help me with X", "absolutely — happy to help with X. Here's a long answer."),
    ])
    merger = JarvisCorpusMerger(
        sources=[CorpusSource(host="mac", remote_path=str(mac_dir), ssh_alias=None)],
        stage_base=tmp_path / "stage",
        corpus_base=tmp_path / "corpus",
        state_dir=tmp_path / "state",
    )
    res = merger.collect()
    assert res["appended"] == 1
    assert res["per_host"]["mac"]["appended"] == 1
    corpus = (tmp_path / "corpus" / "jarvis" / "corpus.jsonl").read_text().strip().split("\n")
    assert len(corpus) == 1
    ex = json.loads(corpus[0])
    assert ex["messages"][1]["content"].startswith("hello there please help me")


def test_collect_namespaces_cursors_by_host(tmp_path: Path):
    """Two hosts with same session UUID must each contribute their own pairs."""
    neb_stage = tmp_path / "stage" / "neb-server"
    trc_stage = tmp_path / "stage" / "trc-server"
    _write_session(neb_stage, "sid-x", [
        ("question one with enough chars to pass", "answer one with enough chars to pass the filter"),
    ])
    _write_session(trc_stage, "sid-x", [
        ("question two with enough chars to pass", "answer two with enough chars to pass the filter"),
    ])

    merger = JarvisCorpusMerger(
        sources=[
            CorpusSource(host="neb-server", remote_path="ignored-bc-staged", ssh_alias="neb-server"),
            CorpusSource(host="trc-server", remote_path="ignored-bc-staged", ssh_alias="trc-server"),
        ],
        stage_base=tmp_path / "stage",
        corpus_base=tmp_path / "corpus",
        state_dir=tmp_path / "state",
    )
    res = merger.collect()
    assert res["appended"] == 2  # both hosts contribute despite shared sid
    assert res["per_host"]["neb-server"]["appended"] == 1
    assert res["per_host"]["trc-server"]["appended"] == 1

    state = json.loads((tmp_path / "state" / "jarvis" / "jarvis-merge-cursors.json").read_text())
    assert "neb-server:sid-x" in state["cursors"]
    assert "trc-server:sid-x" in state["cursors"]


def test_collect_is_idempotent_across_runs(tmp_path: Path):
    stage = tmp_path / "stage" / "neb-server"
    _write_session(stage, "sid-1", [
        ("first question good length here", "first answer good length here for filter pass"),
    ])
    merger = JarvisCorpusMerger(
        sources=[CorpusSource(host="neb-server", remote_path="x", ssh_alias="neb-server")],
        stage_base=tmp_path / "stage",
        corpus_base=tmp_path / "corpus",
        state_dir=tmp_path / "state",
    )
    r1 = merger.collect()
    r2 = merger.collect()
    assert r1["appended"] == 1
    assert r2["appended"] == 0  # cursor caught up
    corpus = (tmp_path / "corpus" / "jarvis" / "corpus.jsonl").read_text().strip().split("\n")
    assert len(corpus) == 1


def test_collect_picks_up_appended_lines_in_existing_session(tmp_path: Path):
    stage = tmp_path / "stage" / "neb-server"
    p = _write_session(stage, "sid-1", [
        ("first user msg good length", "first assistant msg good length"),
    ])
    merger = JarvisCorpusMerger(
        sources=[CorpusSource(host="neb-server", remote_path="x", ssh_alias="neb-server")],
        stage_base=tmp_path / "stage",
        corpus_base=tmp_path / "corpus",
        state_dir=tmp_path / "state",
    )
    merger.collect()

    # Now append a new pair to the same session (simulating live activity)
    with open(p, "a") as f:
        f.write(json.dumps({"type": "message", "message": {"role": "user", "content": "second user msg good length"}, "timestamp": "t1"}) + "\n")
        f.write(json.dumps({"type": "message", "message": {"role": "assistant", "content": "second assistant msg good length"}, "timestamp": "t1a"}) + "\n")

    r2 = merger.collect()
    assert r2["appended"] == 1
    assert r2["per_host"]["neb-server"]["appended"] == 1


def test_collect_filters_short_pairs(tmp_path: Path):
    stage = tmp_path / "stage" / "neb-server"
    _write_session(stage, "sid-short", [
        ("hi", "hello there friend how are you"),       # user too short
        ("longer user msg good length", "k"),            # assistant too short
        ("longer user msg good length", "long enough assistant reply"),  # passes
    ])
    merger = JarvisCorpusMerger(
        sources=[CorpusSource(host="neb-server", remote_path="x", ssh_alias="neb-server")],
        stage_base=tmp_path / "stage",
        corpus_base=tmp_path / "corpus",
        state_dir=tmp_path / "state",
    )
    res = merger.collect()
    assert res["appended"] == 1


def test_collect_skips_missing_dir_without_crash(tmp_path: Path):
    merger = JarvisCorpusMerger(
        sources=[CorpusSource(host="ghost", remote_path=str(tmp_path / "doesnotexist"), ssh_alias=None)],
        stage_base=tmp_path / "stage",
        corpus_base=tmp_path / "corpus",
        state_dir=tmp_path / "state",
    )
    res = merger.collect()
    assert res["appended"] == 0
    assert res["per_host"]["ghost"].get("status") == "no-dir"


def test_collect_ignores_trajectory_files(tmp_path: Path):
    """`*.trajectory.jsonl` are openclaw's tool-trace dumps — not chat msgs."""
    stage = tmp_path / "stage" / "neb-server"
    stage.mkdir(parents=True)
    _write_session(stage, "real", [
        ("good user message length", "good assistant response length"),
    ])
    (stage / "real.trajectory.jsonl").write_text(
        json.dumps({"type": "message", "message": {"role": "user", "content": "should not be in corpus"}}) + "\n"
    )
    merger = JarvisCorpusMerger(
        sources=[CorpusSource(host="neb-server", remote_path="x", ssh_alias="neb-server")],
        stage_base=tmp_path / "stage",
        corpus_base=tmp_path / "corpus",
        state_dir=tmp_path / "state",
    )
    res = merger.collect()
    assert res["appended"] == 1


def test_run_combines_pull_and_collect(tmp_path: Path):
    fake_neb = tmp_path / "fake_neb"
    _write_session(fake_neb, "sid-1", [
        ("real user message long enough", "real assistant response long enough"),
    ])
    runner = _make_runner()
    merger = JarvisCorpusMerger(
        sources=[CorpusSource(host="neb-server", remote_path=str(fake_neb), ssh_alias="neb-server")],
        stage_base=tmp_path / "stage",
        corpus_base=tmp_path / "corpus",
        state_dir=tmp_path / "state",
        rsync_runner=runner,
    )
    out = merger.run()
    assert out["pull"]["neb-server"]["status"] == "pulled"
    assert out["collect"]["appended"] == 1
