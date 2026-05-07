"""ClaudeHarnessSource — port high-signal claude-harness memory entries into HyperSwarm.

claude-harness (panayiotism/claude-harness) is a Claude Code plugin that writes
per-project state under `.claude-harness/` — features, decisions, and procedural
memory (what failed, what worked, what was learned). This Source watches those
files across one or more project roots and emits a HyperSwarm Entry per new id.

Watched files (the high-signal subset):
  - .claude-harness/memory/procedural/failures.json    → "don't repeat this"
  - .claude-harness/memory/procedural/successes.json   → "this approach worked"
  - .claude-harness/memory/learned/*.json              → user-correction rules
  - .claude-harness/features/archive.json              → completed feature lifecycle

Skipped intentionally (low signal-to-noise for cross-runtime sync):
  - episodic/decisions.json (rolling 50, churns)
  - semantic/*.json (per-project context)
  - sessions/{uuid}/* (in-flight)
  - agents/context.json (orchestration plumbing)

How it wires up:

  1. install() walks the configured roots, finds every .claude-harness/ dir,
     and snapshots all currently-present entry ids as "seen". This prevents
     replaying history on first install. Idempotent.

  2. capture(raw) is invoked periodically. It walks the roots again, finds
     the first unseen id, persists it to seen, and returns one Entry. Returns
     None when nothing is new — the orchestrator drains via a loop.

`raw` is intentionally permissive — this source pulls everything it needs from
its config, so the typical invocation passes raw={}.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path

from hyperswarm.core.entry import Entry
from hyperswarm.core.source import Source

DEFAULT_ROOTS = ["~/projects"]
DEFAULT_STATE_PATH = "~/.local/state/hyperswarm/claude_harness.json"
DEFAULT_MAX_DEPTH = 4  # ~/projects/<org>/<project>/.claude-harness


class ClaudeHarnessSource(Source):
    name = "claude_harness"

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        roots_cfg = self.config.get("roots", DEFAULT_ROOTS)
        self.roots = [Path(os.path.expanduser(r)) for r in roots_cfg]
        self.state_path = Path(
            os.path.expanduser(self.config.get("state_path", DEFAULT_STATE_PATH))
        )
        self.max_depth = int(self.config.get("max_depth", DEFAULT_MAX_DEPTH))
        self._runtime_override = self.config.get("runtime_name")

    @property
    def runtime_name(self) -> str:
        return self._runtime_override or self.name

    # ------------------------------------------------------------- install
    def install(self) -> None:
        """Idempotent: snapshot all currently-present entry ids as 'seen'.

        On first install we mark every existing claude-harness entry as already
        seen so we don't flood the store with historical data. Re-running
        install() must not change state.
        """
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        if self.state_path.exists():
            return

        seen: list[str] = []
        for harness_dir in self._find_harness_dirs():
            for keyed_id, _entry, _file_kind in self._iter_entries(harness_dir):
                seen.append(keyed_id)
        self._write_state({"seen_ids": sorted(set(seen))})

    def uninstall(self) -> None:
        if self.state_path.exists():
            self.state_path.unlink()

    # ------------------------------------------------------------- capture
    def capture(self, raw: dict) -> Entry | None:
        seen = set(self._read_state().get("seen_ids", []))

        for harness_dir in self._find_harness_dirs():
            for keyed_id, entry_data, file_kind in self._iter_entries(harness_dir):
                if keyed_id in seen:
                    continue
                # Found a new entry — persist seen and emit
                seen.add(keyed_id)
                self._write_state({"seen_ids": sorted(seen)})
                return self._to_entry(harness_dir, entry_data, file_kind)
        return None

    # ------------------------------------------------------------- helpers
    def _find_harness_dirs(self) -> list[Path]:
        """Walk each root up to max_depth looking for .claude-harness/ dirs."""
        out: list[Path] = []
        for root in self.roots:
            if not root.exists():
                continue
            for candidate in self._walk(root, self.max_depth):
                harness = candidate / ".claude-harness"
                if harness.is_dir():
                    out.append(harness)
        return out

    @staticmethod
    def _walk(root: Path, max_depth: int) -> list[Path]:
        """Yield directories up to max_depth from root (inclusive of root)."""
        result: list[Path] = [root]
        if max_depth <= 0:
            return result
        try:
            for child in root.iterdir():
                if not child.is_dir() or child.name.startswith("."):
                    continue
                result.extend(ClaudeHarnessSource._walk(child, max_depth - 1))
        except OSError:
            pass
        return result

    def _iter_entries(self, harness_dir: Path):
        """Yield (keyed_id, entry_dict, file_kind) for every entry in this harness.

        file_kind is one of: 'failure', 'success', 'learned', 'archive'.
        keyed_id is '<harness_path>::<file_kind>::<entry_id>' so duplicate ids
        across projects don't collide.
        """
        # Procedural failures
        failures_path = harness_dir / "memory" / "procedural" / "failures.json"
        for entry in self._safe_load_entries(failures_path):
            eid = entry.get("id")
            if eid:
                yield f"{harness_dir}::failure::{eid}", entry, "failure"

        # Procedural successes
        successes_path = harness_dir / "memory" / "procedural" / "successes.json"
        for entry in self._safe_load_entries(successes_path):
            eid = entry.get("id")
            if eid:
                yield f"{harness_dir}::success::{eid}", entry, "success"

        # Learned rules — multiple files in learned/
        learned_dir = harness_dir / "memory" / "learned"
        if learned_dir.is_dir():
            for f in sorted(learned_dir.iterdir()):
                if not f.is_file() or f.suffix != ".json":
                    continue
                for entry in self._safe_load_entries(f):
                    eid = entry.get("id")
                    if eid:
                        yield f"{harness_dir}::learned::{eid}", entry, "learned"

        # Archived features
        archive_path = harness_dir / "features" / "archive.json"
        for entry in self._safe_load_archive(archive_path):
            eid = entry.get("id")
            if eid:
                yield f"{harness_dir}::archive::{eid}", entry, "archive"

    @staticmethod
    def _safe_load_entries(path: Path) -> list[dict]:
        """Read a JSON file with shape {entries: [...]}. Tolerate missing/malformed."""
        try:
            data = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return []
        entries = data.get("entries") if isinstance(data, dict) else None
        return entries if isinstance(entries, list) else []

    @staticmethod
    def _safe_load_archive(path: Path) -> list[dict]:
        """archive.json shape: {features: [...], fixes: [...]}. Combine both."""
        try:
            data = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return []
        if not isinstance(data, dict):
            return []
        features = data.get("features") or []
        fixes = data.get("fixes") or []
        out: list[dict] = []
        if isinstance(features, list):
            out.extend(features)
        if isinstance(fixes, list):
            out.extend(fixes)
        return out

    def _to_entry(self, harness_dir: Path, entry_data: dict, file_kind: str) -> Entry:
        project_dir = harness_dir.parent
        eid = entry_data.get("id", "")
        ts = self._parse_timestamp(entry_data.get("timestamp") or entry_data.get("completedAt"))

        summary, body = self._format(entry_data, file_kind)
        return Entry(
            runtime=self.runtime_name,
            cwd=str(project_dir),
            summary=summary,
            body=body,
            session_id=f"{file_kind}-{eid}",
            timestamp=ts,
        )

    @staticmethod
    def _format(entry: dict, kind: str) -> tuple[str, str]:
        eid = entry.get("id", "")
        if kind == "failure":
            approach = entry.get("approach", "")
            summary = f"FAILURE {eid}: {approach}"[:120]
            body_parts = [
                f"# Failure: {eid}",
                "",
                f"- **Feature**: {entry.get('feature', '')}",
                f"- **Approach**: {approach}",
                f"- **Errors**: {', '.join(entry.get('errors') or [])}",
                f"- **Root cause**: {entry.get('rootCause', '')}",
                f"- **Prevention**: {entry.get('prevention', '')}",
            ]
            files = entry.get("files") or []
            if files:
                body_parts.append(f"- **Files**: {', '.join(files)}")
            return summary, "\n".join(body_parts)

        if kind == "success":
            approach = entry.get("approach", "")
            summary = f"SUCCESS {eid}: {approach}"[:120]
            body_parts = [
                f"# Success: {eid}",
                "",
                f"- **Feature**: {entry.get('feature', '')}",
                f"- **Approach**: {approach}",
            ]
            files = entry.get("files") or []
            if files:
                body_parts.append(f"- **Files**: {', '.join(files)}")
            return summary, "\n".join(body_parts)

        if kind == "learned":
            rule = entry.get("rule") or entry.get("description") or ""
            summary = f"LEARNED {eid}: {rule}"[:120]
            body_parts = [f"# Learned rule: {eid}", "", rule]
            why = entry.get("why") or entry.get("reason")
            if why:
                body_parts.extend(["", f"**Why**: {why}"])
            return summary, "\n".join(body_parts)

        if kind == "archive":
            name = entry.get("name", "")
            status = entry.get("status", "")
            summary = f"ARCHIVED {eid} [{status}]: {name}"[:120]
            body_parts = [
                f"# Archived feature: {eid}",
                "",
                f"- **Name**: {name}",
                f"- **Status**: {status}",
                f"- **Description**: {entry.get('description', '')}",
                f"- **Attempts**: {entry.get('attempts', 0)}",
            ]
            return summary, "\n".join(body_parts)

        return f"UNKNOWN {eid}", json.dumps(entry, indent=2)

    @staticmethod
    def _parse_timestamp(ts_raw) -> _dt.datetime:
        if not ts_raw or not isinstance(ts_raw, str):
            return _dt.datetime.now(_dt.timezone.utc)
        try:
            # Handle the 'Z' suffix (Python 3.11+ accepts it; we backfill for 3.9/3.10).
            normalized = ts_raw.replace("Z", "+00:00")
            return _dt.datetime.fromisoformat(normalized)
        except (ValueError, TypeError):
            return _dt.datetime.now(_dt.timezone.utc)

    def _read_state(self) -> dict:
        try:
            return json.loads(self.state_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    def _write_state(self, state: dict) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state))
