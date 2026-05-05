"""ClaudeCodeSource — capture Claude Code sessions via the Stop hook.

How it wires up:

  1. `install()` merges a `Stop` hook into ~/.claude/settings.json that pipes
     the hook's stdin JSON to `hyperswarm capture --runtime claude-code`. The
     hook command also includes `|| true` so a HyperSwarm bug never crashes
     the user's Claude Code session.

  2. When Claude Code finishes responding, it fires the Stop hook with stdin
     JSON like:
        {
          "session_id": "abc123",
          "transcript_path": "/Users/.../.claude/projects/.../abc123.jsonl",
          "cwd": "/Users/.../projects/teamnebula.ai/api",
          "hook_event_name": "Stop"
        }

  3. The CLI calls `capture(raw)` on this Source, which reads the JSONL
     transcript, extracts the latest user prompt + assistant response, and
     returns a populated Entry. Summary defaults to the first 80 chars of
     the user prompt.

Capture is intentionally minimal — extracting "decisions" and "blockers"
from a transcript with any sophistication is a different problem. Phase 2
ships the raw last-turn capture; richer summarization can be a Phase 3
opt-in plugin that wraps this one.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from hyperswarm.core.entry import Entry
from hyperswarm.core.source import Source

DEFAULT_SETTINGS_PATH = "~/.claude/settings.json"
HOOK_MATCHER = ".*"


def _resolve_hyperswarm_binary() -> str:
    """Absolute path to the hyperswarm CLI, falling back to bare name.

    Used at install time so the Stop hook command doesn't depend on
    Claude Code's $PATH at hook-fire time — CC inherits PATH from however
    it was launched (terminal vs Spotlight/Dock), and we can't assume the
    venv's bin dir is there.
    """
    return shutil.which("hyperswarm") or "hyperswarm"


def _build_default_hook_command() -> str:
    binary = _resolve_hyperswarm_binary()
    # Single-quote the path in case it ever contains spaces (HOME paths
    # under "/Users/Some Name/" do).
    return f"'{binary}' capture --runtime claude-code || true"


# Kept for back-compat with anyone importing the module-level constant.
DEFAULT_HOOK_COMMAND = "hyperswarm capture --runtime claude-code || true"


class ClaudeCodeSource(Source):
    name = "claude-code"

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self.settings_path = Path(
            os.path.expanduser(self.config.get("settings_path", DEFAULT_SETTINGS_PATH))
        )
        # User can override via config.toml; otherwise resolve absolute path
        # on the host doing the install so the hook works regardless of how
        # Claude Code is later launched.
        self.hook_command = self.config.get("hook_command") or _build_default_hook_command()

    # ------------------------------------------------------------- install
    def install(self) -> None:
        """Idempotent: re-running this does not duplicate the hook."""
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            settings = json.loads(self.settings_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            settings = {}

        hooks = settings.setdefault("hooks", {})
        stop_hooks = hooks.setdefault("Stop", [])

        # Look for an existing HyperSwarm hook (matched by command substring,
        # so the user can edit the matcher without us deduplicating wrong).
        for entry in stop_hooks:
            for h in entry.get("hooks", []):
                if h.get("type") == "command" and "capture --runtime claude-code" in h.get("command", ""):
                    h["command"] = self.hook_command  # update if version changed
                    self._write(settings)
                    return

        stop_hooks.append({
            "matcher": HOOK_MATCHER,
            "hooks": [{"type": "command", "command": self.hook_command}],
        })
        self._write(settings)

    def uninstall(self) -> None:
        """Remove our Stop hook. No-op if not present."""
        try:
            settings = json.loads(self.settings_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return
        stop_hooks = settings.get("hooks", {}).get("Stop", [])
        kept = []
        for entry in stop_hooks:
            entry_hooks = [
                h for h in entry.get("hooks", [])
                if not (h.get("type") == "command" and "capture --runtime claude-code" in h.get("command", ""))
            ]
            if entry_hooks:
                entry["hooks"] = entry_hooks
                kept.append(entry)
        if stop_hooks:
            settings["hooks"]["Stop"] = kept
        self._write(settings)

    def _write(self, settings: dict) -> None:
        self.settings_path.write_text(json.dumps(settings, indent=2) + "\n")

    # ------------------------------------------------------------- capture
    def capture(self, raw: dict) -> Entry:
        """raw is the Stop hook's stdin JSON — see module docstring for shape.

        Tolerant of missing fields; never raises. A capture that produces a
        nearly-empty Entry is fine; raising would crash the user's session.
        """
        cwd = raw.get("cwd", "") or os.getcwd()
        session_id = raw.get("session_id", "") or ""
        transcript_path = raw.get("transcript_path", "") or ""

        last_user, last_assistant, files_touched = "", "", []
        if transcript_path and Path(transcript_path).exists():
            last_user, last_assistant, files_touched = self._read_transcript(transcript_path)

        summary = self._make_summary(last_user, last_assistant)
        body = self._render_body(last_user, last_assistant, files_touched)

        return Entry(
            runtime=self.name,
            cwd=cwd,
            summary=summary,
            body=body,
            session_id=session_id,
        )

    @staticmethod
    def _read_transcript(path: str) -> tuple[str, str, list[str]]:
        """Return (last_user_text, last_assistant_text, files_touched).

        Reads the JSONL transcript backwards so we don't have to load the
        whole thing for long sessions. JSONL line shape is Claude Code's
        internal session format — best-effort field extraction.
        """
        last_user = ""
        last_assistant = ""
        files: list[str] = []
        try:
            with open(path) as f:
                lines = f.readlines()
        except Exception:
            return last_user, last_assistant, files

        for line in reversed(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            role = row.get("role") or row.get("type") or ""
            text = ClaudeCodeSource._extract_text(row)
            if role == "user" and not last_user and text:
                last_user = text
            elif role == "assistant" and not last_assistant and text:
                last_assistant = text
            # Collect file paths from any tool_use blocks we recognize.
            for path_used in ClaudeCodeSource._extract_file_paths(row):
                if path_used not in files:
                    files.append(path_used)
            if last_user and last_assistant:
                break
        return last_user, last_assistant, files

    @staticmethod
    def _extract_text(row: dict) -> str:
        """Pull a plain-text representation out of a transcript row."""
        msg = row.get("message") or row
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for c in content:
                if not isinstance(c, dict):
                    continue
                if c.get("type") == "text" and isinstance(c.get("text"), str):
                    parts.append(c["text"])
            return "\n".join(parts).strip()
        return ""

    @staticmethod
    def _extract_file_paths(row: dict) -> list[str]:
        """Collect file paths from Edit/Write/Read tool_use blocks."""
        out: list[str] = []
        msg = row.get("message") or row
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            return out
        for c in content:
            if not isinstance(c, dict) or c.get("type") != "tool_use":
                continue
            tool = c.get("name", "")
            inp = c.get("input") or {}
            if tool in ("Edit", "Write", "Read", "MultiEdit") and isinstance(inp.get("file_path"), str):
                out.append(inp["file_path"])
            elif tool == "NotebookEdit" and isinstance(inp.get("notebook_path"), str):
                out.append(inp["notebook_path"])
        return out

    @staticmethod
    def _make_summary(user: str, assistant: str) -> str:
        # Prefer a short slice of the user prompt — that's what a future "what
        # was I doing?" reader cares about. Fall back to assistant text.
        text = user or assistant or "(no transcript text)"
        text = text.replace("\n", " ").strip()
        return text[:80]

    @staticmethod
    def _render_body(user: str, assistant: str, files: list[str]) -> str:
        sections = []
        if user:
            sections.append(f"## User\n\n{user}")
        if assistant:
            sections.append(f"## Assistant\n\n{assistant}")
        if files:
            files_md = "\n".join(f"- {f}" for f in files)
            sections.append(f"## Files touched\n\n{files_md}")
        return "\n\n".join(sections) or "(empty capture)"
