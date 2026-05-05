"""OpenClawRunsSource — capture autonomous OpenClaw agent runs.

OpenClaw stores cron-triggered agent runs as JSONL files in
~/.openclaw/cron/runs/, one file per scheduled job. Each line is a
completed run with metadata: jobId, action, status, summary, error,
sessionId, ts, durationMs, model, provider.

This Source watches that directory and emits one HyperSwarm Entry per
line per file. Cursor model: line-count per file, persisted across
restarts so we never replay or skip.

Why a separate Source from OpenClawSource: OpenClaw has two distinct
storage areas — `~/openclaw-memory/entries/` (manual, populated by the
oc-memory CLI) and `~/.openclaw/cron/runs/` (autonomous, populated by
the agent runner itself). They have different on-disk shapes and
different update patterns. Two sources, one consistent watcher pattern.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path

from hyperswarm.core.entry import Entry
from hyperswarm.core.source import Source

DEFAULT_RUNS_DIR = "~/.openclaw/cron/runs"
DEFAULT_STATE_PATH = "~/.local/state/hyperswarm/openclaw-runs.json"


class OpenClawRunsSource(Source):
    name = "openclaw_runs"

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self.runs_dir = Path(os.path.expanduser(self.config.get("runs_dir", DEFAULT_RUNS_DIR)))
        self.state_path = Path(os.path.expanduser(self.config.get("state_path", DEFAULT_STATE_PATH)))
        self._runtime_override = self.config.get("runtime_name")

    @property
    def runtime_name(self) -> str:
        return self._runtime_override or self.name

    # ------------------------------------------------------------- install
    def install(self) -> None:
        """Seed the cursor at every file's current line count so the first
        capture doesn't replay months of historical runs.
        """
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        if self.state_path.exists():
            return
        cursor: dict[str, int] = {}
        if self.runs_dir.exists():
            for f in self.runs_dir.iterdir():
                if f.is_file() and f.suffix == ".jsonl":
                    cursor[f.name] = _count_lines(f)
        self._write_state(cursor)

    def uninstall(self) -> None:
        if self.state_path.exists():
            self.state_path.unlink()

    # ------------------------------------------------------------- capture
    def capture(self, raw: dict) -> Entry | None:
        """Return an Entry for the next unseen line in any jsonl file, or None.

        Drained iteratively by callers — the cron tick should call until None.
        """
        cursors = self._read_state()
        if not self.runs_dir.exists():
            return None

        for f in sorted(self.runs_dir.iterdir()):
            if not f.is_file() or f.suffix != ".jsonl":
                continue
            seen = cursors.get(f.name, 0)
            try:
                with open(f) as fp:
                    lines = fp.readlines()
            except OSError:
                continue
            if seen >= len(lines):
                continue
            line = lines[seen]
            cursors[f.name] = seen + 1
            self._write_state(cursors)
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # Malformed line — advance past it and let the caller try again.
                return None
            return self._line_to_entry(record)
        return None

    def _line_to_entry(self, record: dict) -> Entry:
        ts_ms = record.get("ts") or record.get("runAtMs") or 0
        if ts_ms:
            ts = _dt.datetime.fromtimestamp(ts_ms / 1000, tz=_dt.timezone.utc)
        else:
            ts = _dt.datetime.now(tz=_dt.timezone.utc)

        action = record.get("action") or "?"
        status = record.get("status") or "?"
        summary_text = record.get("summary") or record.get("error") or f"{action} ({status})"
        summary = str(summary_text).splitlines()[0][:80] if summary_text else f"{action} ({status})"

        lines = [f"# OpenClaw run: {action} ({status})", ""]
        for k in ("jobId", "sessionId", "sessionKey", "action", "status",
                  "durationMs", "model", "provider", "deliveryStatus"):
            if record.get(k):
                lines.append(f"- **{k}**: {record[k]}")
        if record.get("error"):
            lines.append("")
            lines.append(f"## Error\n\n{record['error']}")
        if record.get("summary") and record.get("summary") != summary:
            lines.append("")
            lines.append(f"## Summary\n\n{record['summary']}")

        return Entry(
            runtime=self.runtime_name,
            cwd=str(self.runs_dir),
            summary=summary,
            body="\n".join(lines),
            session_id=str(record.get("sessionId") or record.get("jobId") or ""),
            timestamp=ts,
        )

    # ------------------------------------------------------------- state
    def _read_state(self) -> dict:
        try:
            return json.loads(self.state_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _write_state(self, state: dict) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state))


def _count_lines(path: Path) -> int:
    try:
        with open(path) as f:
            return sum(1 for _ in f)
    except OSError:
        return 0
