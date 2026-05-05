"""OpenClawSource — port OpenClaw session entries into HyperSwarm.

OpenClaw already writes append-only entries to a directory on its host. This
Source watches that directory and converts each new entry into a HyperSwarm
Entry, idempotently.

How it wires up:

  1. `install()` writes a small state file at <state_path> (default
     ~/.local/state/hyperswarm/openclaw.json) recording the cursor (highest
     mtime seen). On first install, the cursor starts at "now" so we don't
     replay the entire historical archive on first capture.

  2. `capture(raw)` is invoked periodically (cron, systemd timer, or
     `hyperswarm capture --runtime openclaw` from a daemon). It scans the
     watch_dir for files modified after the cursor, ports each into an
     Entry, and advances the cursor.

`raw` is intentionally permissive — this source pulls everything it needs
from its config, so the typical invocation passes `raw={}`.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from hyperswarm.core.entry import Entry
from hyperswarm.core.source import Source

DEFAULT_WATCH_DIR = "~/openclaw-memory/entries"
DEFAULT_STATE_PATH = "~/.local/state/hyperswarm/openclaw.json"


class OpenClawSource(Source):
    name = "openclaw"

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self.watch_dir = Path(
            os.path.expanduser(self.config.get("watch_dir", DEFAULT_WATCH_DIR))
        )
        self.state_path = Path(
            os.path.expanduser(self.config.get("state_path", DEFAULT_STATE_PATH))
        )
        # Optional config knob — lets users tag entries from this source with
        # an explicit runtime name (e.g. "openclaw-neb" vs the default "openclaw")
        # without writing a subclass.
        self._runtime_override = self.config.get("runtime_name")

    @property
    def runtime_name(self) -> str:
        return self._runtime_override or self.name

    # ------------------------------------------------------------- install
    def install(self) -> None:
        """Idempotent: initialises the cursor to now() if no state exists yet.

        Existing state is left alone — re-running install() must not replay
        history we've already captured.
        """
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        if self.state_path.exists():
            return
        self._write_state({"cursor_mtime": time.time()})

    def uninstall(self) -> None:
        """Remove our cursor state. Safe to re-install afterwards."""
        if self.state_path.exists():
            self.state_path.unlink()

    # ------------------------------------------------------------- capture
    def capture(self, raw: dict) -> Entry | None:
        """Returns one Entry for the most-recent unseen OpenClaw file, or
        None if nothing new is available.

        The orchestrator should call capture() in a loop until it returns
        None to drain a backlog (the alternative — returning a list — would
        force every Source to handle the backlog idea, which most don't have).
        """
        cursor = self._read_cursor()
        new_files = self._unseen_files(cursor)
        if not new_files:
            return None

        # Process the oldest unseen file first, advance cursor.
        target = new_files[0]
        try:
            mtime = target.stat().st_mtime
            text = target.read_text(errors="replace")
        except Exception:
            # Skip a malformed file by advancing past it; don't get stuck.
            self._write_cursor(target.stat().st_mtime if target.exists() else cursor + 1)
            return None

        self._write_cursor(mtime)
        return Entry(
            runtime=self.runtime_name,
            cwd=str(self.watch_dir),
            summary=self._first_line(text),
            body=text.strip(),
            session_id=target.stem,
            timestamp=self._mtime_to_dt(mtime),
        )

    # ------------------------------------------------------------- helpers
    def _read_cursor(self) -> float:
        try:
            data = json.loads(self.state_path.read_text())
            return float(data.get("cursor_mtime", 0))
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            return 0.0

    def _write_cursor(self, mtime: float) -> None:
        self._write_state({"cursor_mtime": mtime})

    def _write_state(self, state: dict) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state))

    def _unseen_files(self, cursor: float) -> list[Path]:
        if not self.watch_dir.exists():
            return []
        out: list[Path] = []
        for f in self.watch_dir.iterdir():
            if not f.is_file():
                continue
            try:
                if f.stat().st_mtime > cursor:
                    out.append(f)
            except OSError:
                continue
        # Process oldest-first so the cursor advances monotonically.
        return sorted(out, key=lambda p: p.stat().st_mtime)

    @staticmethod
    def _first_line(text: str, max_len: int = 80) -> str:
        """Prefer the entry's markdown heading (that's where OpenClaw puts the
        session title), fall back to first non-empty body line."""
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                return stripped.lstrip("# ").strip()[:max_len] or "(untitled)"
        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped[:max_len]
        return "(empty openclaw entry)"

    @staticmethod
    def _mtime_to_dt(mtime: float):
        import datetime as _dt
        return _dt.datetime.fromtimestamp(mtime, tz=_dt.timezone.utc)
