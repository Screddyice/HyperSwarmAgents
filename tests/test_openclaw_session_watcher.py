"""Tests for OpenClawSessionWatcher — the event-driven daemon that fires
reflect/tune-collect/tune-trigger on session-end.

The watcher is a coordinator (no LLM calls of its own), so tests focus on:
- which sessions get marked ready (mtime + debounce logic)
- the right CLI subprocess calls fire in the right order (reflect → tune-collect;
  training is NOT auto-fired because it requires CUDA which the watcher host
  likely lacks)
- already-processed sessions don't re-fire until there's new activity
- failure of one CLI call doesn't break the loop
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from hyperswarm.watchers.openclaw_sessions import OpenClawSessionWatcher


def _touch_session(tmp_path: Path, agent: str, sid: str, mtime: float) -> Path:
    sdir = tmp_path / "agents" / agent / "sessions"
    sdir.mkdir(parents=True, exist_ok=True)
    p = sdir / f"{sid}.jsonl"
    p.write_text("dummy\n")
    os.utime(p, (mtime, mtime))
    return p


def test_scan_picks_idle_sessions_only(tmp_path: Path):
    now = time.time()
    # Idle session (last write 600s ago — past 300s debounce)
    _touch_session(tmp_path, "jarvis", "old-idle", mtime=now - 600)
    # Recent session (last write 60s ago — under debounce)
    _touch_session(tmp_path, "jarvis", "still-active", mtime=now - 60)

    w = OpenClawSessionWatcher(
        agents=["jarvis"],
        agents_dir=tmp_path / "agents",
        debounce_s=300,
        hyperswarm_bin="/bin/true",  # dummy
    )
    ready = w._scan_once()
    assert ("jarvis", "old-idle") in ready
    assert ("jarvis", "still-active") not in ready


def test_scan_ignores_trajectory_files(tmp_path: Path):
    now = time.time()
    sdir = tmp_path / "agents" / "jarvis" / "sessions"
    sdir.mkdir(parents=True)
    real = sdir / "abc.jsonl"
    real.write_text("dummy\n")
    os.utime(real, (now - 600, now - 600))
    traj = sdir / "abc.trajectory.jsonl"
    traj.write_text("dummy\n")
    os.utime(traj, (now - 600, now - 600))

    w = OpenClawSessionWatcher(
        agents=["jarvis"], agents_dir=tmp_path / "agents", debounce_s=300, hyperswarm_bin="/bin/true"
    )
    ready = w._scan_once()
    assert ("jarvis", "abc") in ready
    assert ("jarvis", "abc.trajectory") not in ready


def test_already_processed_does_not_refire(tmp_path: Path):
    now = time.time()
    _touch_session(tmp_path, "jarvis", "session-a", mtime=now - 600)
    w = OpenClawSessionWatcher(
        agents=["jarvis"], agents_dir=tmp_path / "agents", debounce_s=300, hyperswarm_bin="/bin/true"
    )
    ready1 = w._scan_once()
    assert ("jarvis", "session-a") in ready1
    # Mark as processed (what _process does)
    w._state[("jarvis", "session-a")].last_processed_mtime = w._state[
        ("jarvis", "session-a")
    ].last_mtime

    # Second scan: same session, no new mtime → not ready
    ready2 = w._scan_once()
    assert ready2 == []


def test_processed_session_refires_when_mtime_advances(tmp_path: Path):
    """If the agent resumes a session (writes more lines later), the watcher
    should re-trigger after the next idle window."""
    now = time.time()
    p = _touch_session(tmp_path, "jarvis", "session-b", mtime=now - 600)
    w = OpenClawSessionWatcher(
        agents=["jarvis"], agents_dir=tmp_path / "agents", debounce_s=300, hyperswarm_bin="/bin/true"
    )
    # First pass — ready, mark processed
    ready1 = w._scan_once()
    assert ("jarvis", "session-b") in ready1
    w._state[("jarvis", "session-b")].last_processed_mtime = w._state[
        ("jarvis", "session-b")
    ].last_mtime

    # Bump mtime forward (still idle for 600s — pretend session resumed
    # and went idle again). Forward enough that processed_mtime < new_mtime.
    new_mtime = now - 400
    os.utime(p, (new_mtime, new_mtime))

    ready2 = w._scan_once()
    assert ("jarvis", "session-b") in ready2


def test_process_fires_three_subprocess_calls_in_order(tmp_path: Path, monkeypatch):
    calls: list[list[str]] = []

    class FakeResult:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return FakeResult()

    monkeypatch.setattr("subprocess.run", fake_run)

    w = OpenClawSessionWatcher(
        agents=["jarvis"],
        agents_dir=tmp_path / "agents",
        hyperswarm_bin="/usr/bin/hyperswarm",
        enable_tune=True,
    )
    # Need to simulate state so _process knows the session
    now = time.time()
    _touch_session(tmp_path, "jarvis", "session-c", mtime=now - 600)
    w._scan_once()  # populates _state

    w._process("jarvis", "session-c")

    # 2 calls: reflect, tune-collect (training is manual on a GPU host, not
    # auto-fired by the watcher)
    assert len(calls) == 2
    assert calls[0] == ["/usr/bin/hyperswarm", "reflect", "--agent", "jarvis"]
    assert calls[1] == ["/usr/bin/hyperswarm", "tune-collect", "--agent", "jarvis"]
    # processed_mtime advances
    s = w._state[("jarvis", "session-c")]
    assert s.last_processed_mtime == s.last_mtime


def test_process_with_no_tune_skips_tune_calls(tmp_path: Path, monkeypatch):
    calls: list[list[str]] = []

    class FakeResult:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr("subprocess.run", lambda cmd, **kw: (calls.append(cmd) or FakeResult()))

    w = OpenClawSessionWatcher(
        agents=["jarvis"],
        agents_dir=tmp_path / "agents",
        hyperswarm_bin="/usr/bin/hyperswarm",
        enable_tune=False,
    )
    now = time.time()
    _touch_session(tmp_path, "jarvis", "x", mtime=now - 600)
    w._scan_once()
    w._process("jarvis", "x")
    assert len(calls) == 1
    assert calls[0][1] == "reflect"


def test_failed_cli_call_does_not_crash_loop(tmp_path: Path, monkeypatch, caplog):
    class FakeResult:
        returncode = 1
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr("subprocess.run", lambda cmd, **kw: FakeResult())

    w = OpenClawSessionWatcher(
        agents=["jarvis"], agents_dir=tmp_path / "agents", hyperswarm_bin="/usr/bin/hyperswarm"
    )
    now = time.time()
    _touch_session(tmp_path, "jarvis", "y", mtime=now - 600)
    w._scan_once()
    # Should not raise, just log a warning per failed call
    w._process("jarvis", "y")
