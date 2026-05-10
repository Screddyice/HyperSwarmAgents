"""OpenClaw session watcher — event-driven trigger for reflection + tuning.

Replaces the per-agent cron with a long-running daemon (one per server,
managed by systemd) that polls session jsonl mtimes, detects "session
went idle for N seconds" events, and fires:

  1. `hyperswarm reflect --agent <id>` (distill into curated memories)
  2. `hyperswarm tune-collect --agent <id>` (append to fine-tune corpus)

The watcher does NOT auto-fire `tune-train-local` because training requires
a CUDA GPU which the watcher's host typically lacks. Training is a manual
step on a separate GPU host that pulls the corpus, trains a LoRA adapter,
and writes the adapter back.

Why poll instead of inotify: poll keeps the dependency surface small
(stdlib only), latency tolerance is high (idle detection is debounced
to ~5 minutes anyway), and one process per server is cheaper than
per-agent inotify watchers. ~10s poll cadence × cheap stat call = sub-1%
CPU overhead.

Cost discipline:
- Reflect/tune-collect have idempotent cursors — re-firing is free.
- LLM calls in reflect happen only if new turns since last cursor.
- Fine-tune trigger gates on min_new_examples threshold.
- The watcher itself never calls an LLM directly; it's a coordinator.
"""
from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_AGENTS_DIR = "~/.openclaw/agents"
DEFAULT_POLL_INTERVAL_S = 30
DEFAULT_DEBOUNCE_S = 300  # 5 min idle = "session ended"


def _expand(p: str | os.PathLike) -> Path:
    return Path(os.path.expanduser(str(p)))


@dataclass
class _SessionState:
    """Per-session-jsonl tracking for the watcher."""

    last_mtime: float = 0.0  # most recent mtime we observed
    last_processed_mtime: float = 0.0  # mtime at which we last triggered reflect/tune


@dataclass
class OpenClawSessionWatcher:
    agents: list[str]
    host: str = ""
    agents_dir: Path | None = None
    poll_interval_s: int = DEFAULT_POLL_INTERVAL_S
    debounce_s: int = DEFAULT_DEBOUNCE_S
    hyperswarm_bin: str = ""
    enable_tune: bool = True
    log: logging.Logger = field(default_factory=lambda: logging.getLogger("hyperswarm.watch"))
    _state: dict[tuple[str, str], _SessionState] = field(default_factory=dict, init=False)
    _running: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        self.host = self.host or socket.gethostname()
        self.agents_dir = _expand(self.agents_dir or DEFAULT_AGENTS_DIR)
        if not self.hyperswarm_bin:
            self.hyperswarm_bin = os.environ.get("HYPERSWARM_BIN") or self._discover_bin()

    @staticmethod
    def _discover_bin() -> str:
        # Prefer a venv install adjacent to the package, else $PATH lookup
        for cand in (
            Path.home() / ".hyperswarm" / "venv" / "bin" / "hyperswarm",
            Path.home() / ".local" / "bin" / "hyperswarm",
        ):
            if cand.exists():
                return str(cand)
        return "hyperswarm"

    def stop(self) -> None:
        self._running = False

    def _scan_once(self) -> list[tuple[str, str]]:
        """Return list of (agent, session_id) tuples whose jsonl has been
        idle (no writes) for >= debounce_s AND has new mtime since last
        processing. These are the sessions ready to reflect/tune on."""
        now = time.time()
        ready: list[tuple[str, str]] = []
        for agent in self.agents:
            sdir = self.agents_dir / agent / "sessions"
            if not sdir.exists():
                continue
            for jsonl in sdir.glob("*.jsonl"):
                if jsonl.name.endswith(".trajectory.jsonl"):
                    continue
                sid = jsonl.stem
                key = (agent, sid)
                state = self._state.get(key) or _SessionState()
                try:
                    mtime = jsonl.stat().st_mtime
                except FileNotFoundError:
                    continue
                state.last_mtime = mtime
                self._state[key] = state
                # Ready if: new content since we last processed, AND idle for debounce_s
                if (
                    mtime > state.last_processed_mtime
                    and now - mtime >= self.debounce_s
                ):
                    ready.append(key)
        return ready

    def _run_cli(self, *args: str) -> int:
        """Invoke `hyperswarm <args>` as a subprocess. Returns exit code."""
        cmd = [self.hyperswarm_bin, *args]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            self.log.warning("watch: %s timed out", " ".join(cmd))
            return 124
        if r.returncode != 0:
            self.log.warning(
                "watch: %s exited %d: %s", " ".join(cmd), r.returncode, r.stderr.strip()[:400]
            )
        else:
            self.log.info("watch: %s ok: %s", " ".join(cmd), r.stdout.strip()[:200])
        return r.returncode

    def _process(self, agent: str, sid: str) -> None:
        """Fire reflect + tune-collect for one ready (agent, sid).

        Note: actual LoRA training (`tune-train-local`) is NOT auto-fired by
        the watcher because it requires a CUDA host that the watcher's host
        likely isn't. Training is run manually on a GPU box that pulls the
        corpus, trains, and writes the resulting adapter back. The watcher's
        job stops at corpus collection.
        """
        # Reflect (distill new turns into curated memories)
        self._run_cli("reflect", "--agent", agent)
        if self.enable_tune:
            # Append user/assistant pairs to fine-tune corpus
            self._run_cli("tune-collect", "--agent", agent)
        # Mark this session's mtime as processed so we don't re-trigger until
        # there's MORE activity past this point.
        key = (agent, sid)
        if key in self._state:
            self._state[key].last_processed_mtime = self._state[key].last_mtime

    def loop(self) -> None:
        """Main loop. Runs until stop() or process exit."""
        self.log.info(
            "watcher started: agents=%s host=%s poll=%ds debounce=%ds tune=%s",
            self.agents,
            self.host,
            self.poll_interval_s,
            self.debounce_s,
            self.enable_tune,
        )
        while self._running:
            try:
                ready = self._scan_once()
            except Exception as e:
                self.log.exception("watch: scan failed: %s", e)
                ready = []
            for agent, sid in ready:
                if not self._running:
                    break
                try:
                    self._process(agent, sid)
                except Exception as e:
                    self.log.exception("watch: process failed for %s/%s: %s", agent, sid, e)
            for _ in range(self.poll_interval_s):
                if not self._running:
                    break
                time.sleep(1)


def run_watcher(agents: list[str], **kwargs) -> int:
    """Top-level entry. Configures logging and runs the loop."""
    logging.basicConfig(
        level=os.environ.get("HYPERSWARM_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    OpenClawSessionWatcher(agents=agents, **kwargs).loop()
    return 0
