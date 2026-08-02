"""Mem0SessionSource — session-level, significance-gated distillation from Mem0 Platform.

Replaces ClaudeMemSessionSource after claude-mem was retired (2026-08-02).
Same corpus contract: at most one distilled Entry per session, only when the
session has a course-change / missed-pr / lesson, or a deterministic left-off
handoff when there is substantive unfinished work.

Data sources (read-only against Mem0 Platform; never mutates the store):
  1. Mem0 memories for user_id (default screddy), preferring
     metadata.type=session_summary / session_state and metadata.session_id match
  2. Optional Claude/Codex transcript_path on the hook payload for user_prompt
     and files_edited (same fields the old claude-mem overview carried)

Wiring:
  install() merges a SessionEnd hook that pipes stdin to
  `hyperswarm capture --runtime mem0_session`.
  Left-off wrappers resolve the most recent session_id via Mem0 (or hook stdin).

Runtime name is "mem0-session" — not on the corpus exclude denylist (only
claude-mem-recall is excluded). Body format keeps ## Trigger / ## Lesson /
## User / ## Session so shawn-corpus prefilter still accepts distilled entries.
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

from hyperswarm.core.entry import Entry
from hyperswarm.core.source import Source
from hyperswarm.sources.claude_mem_session import (
    TRIGGER_COURSE_CHANGE,
    TRIGGER_LEFTOFF,
    TRIGGER_LESSON,
    TRIGGER_MISSED_PR,
    _BOILERPLATE_NEXT_STEPS,
    _GATE_SYSTEM_PROMPT,
    _parse_gate_json,
)

DEFAULT_SETTINGS_PATH = "~/.claude/settings.json"
DEFAULT_USER_ID = "screddy"
HOOK_MATCHER = ".*"
HOOK_SUBSTRING = "capture --runtime mem0_session"
_VALID_TRIGGERS = {TRIGGER_COURSE_CHANGE, TRIGGER_MISSED_PR, TRIGGER_LESSON}

_PROJECTS_ENV = Path.home() / "projects" / ".env"


def _resolve_hyperswarm_binary() -> str:
    return shutil.which("hyperswarm") or "hyperswarm"


def _build_default_hook_command() -> str:
    binary = _resolve_hyperswarm_binary()
    return f"'{binary}' capture --runtime mem0_session || true"


def _load_api_key() -> str:
    key = (os.environ.get("MEM0_API_KEY") or "").strip()
    if key.startswith("m0-"):
        return key
    if _PROJECTS_ENV.is_file():
        try:
            for line in _PROJECTS_ENV.read_text().splitlines():
                line = line.strip()
                if line.startswith("MEM0_API_KEY="):
                    v = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if v.startswith("m0-"):
                        return v
        except OSError:
            pass
    return ""


def _parse_created_at(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        # Mem0 returns ISO with offset; normalize Z
        cleaned = value.replace("Z", "+00:00")
        return datetime.fromisoformat(cleaned)
    except (TypeError, ValueError):
        return None


class Mem0SessionSource(Source):
    name = "mem0-session"

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self.settings_path = Path(
            os.path.expanduser(self.config.get("settings_path", DEFAULT_SETTINGS_PATH))
        )
        self.user_id = self.config.get("user_id") or os.environ.get(
            "MEM0_USER_ID", DEFAULT_USER_ID
        )
        self.lookback_hours = int(self.config.get("lookback_hours", 6))
        self.hook_command = self.config.get("hook_command") or _build_default_hook_command()
        self.hook_event = self.config.get("hook_event", "SessionEnd")
        self.gate_model = self.config.get("gate_model", "gpt-5.4")
        self.gate_timeout = int(self.config.get("gate_timeout", 60))
        self._gate_fn = self.config.get("gate_fn")
        # Test seam: list of memory dicts instead of live Platform API
        self._memories_fn = self.config.get("memories_fn")

    # ------------------------------------------------------------- install
    def install(self) -> None:
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            settings = json.loads(self.settings_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            settings = {}

        hooks = settings.setdefault("hooks", {})
        event_hooks = hooks.setdefault(self.hook_event, [])

        for entry in event_hooks:
            for h in entry.get("hooks", []):
                if h.get("type") == "command" and HOOK_SUBSTRING in h.get("command", ""):
                    h["command"] = self.hook_command
                    self._write(settings)
                    return
                # Migrate retired claude-mem capture hook in place
                cmd = h.get("command", "")
                if h.get("type") == "command" and "capture --runtime claude_mem_session" in cmd:
                    h["command"] = self.hook_command
                    self._write(settings)
                    return

        event_hooks.append({
            "matcher": HOOK_MATCHER,
            "hooks": [{"type": "command", "command": self.hook_command}],
        })
        self._write(settings)

    def uninstall(self) -> None:
        try:
            settings = json.loads(self.settings_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return
        changed = False
        for event in (self.hook_event, "SessionEnd", "Stop"):
            event_hooks = settings.get("hooks", {}).get(event, [])
            if not event_hooks:
                continue
            kept = []
            for entry in event_hooks:
                entry_hooks = [
                    h for h in entry.get("hooks", [])
                    if not (
                        h.get("type") == "command"
                        and (
                            HOOK_SUBSTRING in h.get("command", "")
                            or "capture --runtime claude_mem_session" in h.get("command", "")
                        )
                    )
                ]
                if entry_hooks:
                    entry["hooks"] = entry_hooks
                    kept.append(entry)
            settings["hooks"][event] = kept
            changed = True
        if changed:
            self._write(settings)

    def _write(self, settings: dict) -> None:
        self.settings_path.write_text(json.dumps(settings, indent=2) + "\n")

    # ------------------------------------------------------------- capture
    def capture(self, raw: dict) -> Entry | None:
        """raw is SessionEnd/Stop/ManualPush hook JSON. Never raises."""
        try:
            return self._capture_inner(raw or {})
        except Exception:
            return None

    def _capture_inner(self, raw: dict) -> Entry | None:
        cwd = raw.get("cwd", "") or os.getcwd()
        content_session_id = (
            raw.get("session_id")
            or raw.get("sessionId")
            or ""
        )
        if not content_session_id:
            return None

        overview = self._build_overview(content_session_id, cwd, raw)
        if overview is None:
            return None

        if not self._structurally_significant(overview):
            return None

        gate = self._run_gate(overview)
        if not gate.get("qualifies"):
            return self._leftoff_entry(overview, cwd)

        memory_session_id = overview["memory_session_id"]
        summary = gate.get("headline") or self._fallback_headline(overview)
        body = self._render_body(gate, overview)

        return Entry(
            runtime=self.name,
            cwd=cwd,
            summary=summary,
            body=body,
            session_id=memory_session_id,
            project=overview.get("project") or "",
        )

    # ------------------------------------------------------------- mem0 + overview
    def _fetch_memories(self, session_id: str) -> list[dict]:
        if self._memories_fn is not None:
            return list(self._memories_fn(session_id) or [])

        api_key = _load_api_key()
        if not api_key:
            return []

        try:
            from mem0 import MemoryClient
        except ImportError:
            return []

        client = MemoryClient(api_key=api_key)
        results: list[dict] = []

        # Prefer session-scoped summaries first
        for filters in (
            {
                "AND": [
                    {"user_id": self.user_id},
                    {"metadata": {"session_id": session_id}},
                ]
            },
            {
                "AND": [
                    {"user_id": self.user_id},
                    {"metadata": {"type": "session_summary"}},
                ]
            },
            {"user_id": self.user_id},
        ):
            try:
                resp = client.get_all(filters=filters, page_size=50)
            except Exception:
                continue
            batch = resp.get("results") if isinstance(resp, dict) else resp
            if not batch:
                continue
            results.extend(batch if isinstance(batch, list) else [])
            # session_id match is enough; stop early
            if "session_id" in str(filters):
                break

        return self._dedupe_memories(results)

    @staticmethod
    def _dedupe_memories(memories: list[dict]) -> list[dict]:
        seen: set[str] = set()
        out: list[dict] = []
        for m in memories:
            mid = str(m.get("id") or m.get("memory") or "")
            if not mid or mid in seen:
                continue
            seen.add(mid)
            out.append(m)
        return out

    def _build_overview(self, session_id: str, cwd: str, raw: dict) -> dict | None:
        memories = self._fetch_memories(session_id)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.lookback_hours)

        session_mems: list[dict] = []
        recent_mems: list[dict] = []
        for m in memories:
            meta = m.get("metadata") or {}
            if not isinstance(meta, dict):
                meta = {}
            if meta.get("session_id") == session_id:
                session_mems.append(m)
                continue
            created = _parse_created_at(m.get("created_at") or m.get("updated_at"))
            if created is not None:
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                if created >= cutoff:
                    recent_mems.append(m)
            else:
                recent_mems.append(m)

        chosen = session_mems or recent_mems
        # If neither session match nor recent window, still allow overview when
        # we have ANY session_summary text — otherwise None (routine / empty).
        if not chosen and not memories:
            return None
        if not chosen:
            # Global facts only, no recency — too noisy for a session entry
            return None

        learned_lines = []
        for m in chosen[:25]:
            text = (m.get("memory") or m.get("text") or "").strip()
            if text:
                learned_lines.append(f"- {text}")
        learned = "\n".join(learned_lines)

        project = ""
        if cwd:
            project = Path(cwd).name
        for m in chosen:
            meta = m.get("metadata") or {}
            if isinstance(meta, dict) and meta.get("project"):
                project = str(meta["project"])
                break

        user_prompt, files_edited, transcript_request = self._transcript_bits(raw)
        request = transcript_request or (
            f"Coding session in {project or cwd or 'workspace'}"
        )

        next_steps = self._infer_next_steps(chosen, learned, cwd)

        return {
            "memory_session_id": f"mem0-{session_id}",
            "content_session_id": session_id,
            "project": project or "",
            "user_prompt": user_prompt,
            "status": "completed",
            "request": request,
            "investigated": "",
            "learned": learned,
            "completed": "",
            "next_steps": next_steps,
            "files_read": "",
            "files_edited": files_edited,
            "notes": f"mem0_facts={len(chosen)} lookback_h={self.lookback_hours}",
            "observation_count": len(chosen),
        }

    def _transcript_bits(self, raw: dict) -> tuple[str, str, str]:
        """Best-effort user_prompt, files_edited, request from transcript_path."""
        path = raw.get("transcript_path") or raw.get("transcriptPath") or ""
        if not path or not os.path.isfile(path):
            return "", "", ""
        try:
            with open(path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 400_000))
                data = f.read().decode("utf-8", errors="replace")
        except OSError:
            return "", "", ""

        user_prompt = ""
        files: set[str] = set()
        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Claude Code JSONL shapes vary
            role = None
            text = ""
            msg = entry.get("message") or entry
            if isinstance(msg, dict):
                role = msg.get("role") or entry.get("type")
                content = msg.get("content")
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    parts = []
                    for block in content:
                        if isinstance(block, str):
                            parts.append(block)
                        elif isinstance(block, dict):
                            if block.get("type") == "text":
                                parts.append(block.get("text") or "")
                            if block.get("type") == "tool_use":
                                inp = block.get("input") or {}
                                if isinstance(inp, dict) and inp.get("file_path"):
                                    files.add(str(inp["file_path"]))
                    text = "\n".join(parts)
            if role in ("user", "human") and text and not user_prompt:
                # first user message
                cleaned = text.strip()
                if cleaned and not cleaned.startswith("<"):
                    user_prompt = cleaned[:2000]
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            inp = block.get("input") or {}
                            if isinstance(inp, dict) and inp.get("file_path"):
                                files.add(str(inp["file_path"]))

        files_edited = ", ".join(sorted(files)[:20])
        request = user_prompt.split("\n", 1)[0][:200] if user_prompt else ""
        return user_prompt, files_edited, request

    @staticmethod
    def _infer_next_steps(memories: list[dict], learned: str, cwd: str) -> str:
        """Synthesize handoff text when Mem0 facts imply unfinished work."""
        hints: list[str] = []
        for m in memories:
            text = (m.get("memory") or "").strip().lower()
            if any(
                k in text
                for k in (
                    "next step",
                    "todo",
                    "still need",
                    "unfinished",
                    "left off",
                    "continue",
                    "blocked",
                    "wip",
                    "in progress",
                )
            ):
                hints.append((m.get("memory") or "").strip())
        if hints:
            return " ".join(hints[:3])[:500]
        if learned and len(learned) >= 40:
            proj = Path(cwd).name if cwd else "project"
            return (
                f"Resume work in {proj} using recent Mem0 context "
                f"({len(memories)} fact(s) from this session window)."
            )
        return ""

    # ------------------------------------------------------------- gating
    @staticmethod
    def _structurally_significant(overview: dict) -> bool:
        teaching = " ".join(
            (overview.get(k) or "") for k in ("learned", "notes", "next_steps")
        ).strip()
        return len(teaching) >= 12

    def _run_gate(self, overview: dict) -> dict:
        prompt = (
            _GATE_SYSTEM_PROMPT.replace("distilled by claude-mem", "distilled from Mem0 Platform")
            + "\n\nSESSION OVERVIEW:\n"
            + self._overview_text(overview)
        )
        try:
            if self._gate_fn is not None:
                raw = self._gate_fn(prompt)
            else:
                raw = self._call_codex(prompt)
        except Exception:
            return {
                "qualifies": False,
                "trigger": None,
                "headline": "",
                "lesson_body": "",
                "reason": "gate_error",
            }
        return _parse_gate_json(raw)

    def _call_codex(self, prompt: str) -> str:
        try:
            from shawn_corpus.consolidation.codex_client import call_codex  # type: ignore

            return call_codex(prompt, model=self.gate_model, timeout=self.gate_timeout)
        except ImportError:
            return self._call_codex_local(prompt)

    def _call_codex_local(self, prompt: str) -> str:
        import subprocess
        import tempfile

        codex_bin = shutil.which("codex") or os.path.expanduser("~/.npm-global/bin/codex")
        with tempfile.NamedTemporaryFile(mode="r", suffix=".txt", delete=False) as tf:
            out_path = Path(tf.name)
        try:
            result = subprocess.run(
                [
                    codex_bin,
                    "exec",
                    "--model",
                    self.gate_model,
                    "--color",
                    "never",
                    "--skip-git-repo-check",
                    "--output-last-message",
                    str(out_path),
                ],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.gate_timeout,
                env={**os.environ, "CODEX_NO_INTERACTIVE": "1"},
            )
            if result.returncode != 0:
                return ""
            return out_path.read_text()
        finally:
            try:
                out_path.unlink()
            except OSError:
                pass

    # ------------------------------------------------------------- render
    @staticmethod
    def _overview_text(overview: dict) -> str:
        parts = []
        for label, key in (
            ("request", "request"),
            ("investigated", "investigated"),
            ("learned", "learned"),
            ("completed", "completed"),
            ("next_steps", "next_steps"),
            ("notes", "notes"),
        ):
            v = (overview.get(key) or "").strip()
            if v:
                parts.append(f"{label}: {v}")
        return "\n".join(parts)

    @staticmethod
    def _fallback_headline(overview: dict) -> str:
        text = overview.get("request") or overview.get("learned") or "(mem0 session)"
        return text.replace("\n", " ").strip()[:100]

    @staticmethod
    def _substantive_next_steps(overview: dict) -> str:
        text = (overview.get("next_steps") or "").strip()
        if len(text) < 12:
            return ""
        if _BOILERPLATE_NEXT_STEPS.match(text):
            return ""
        return text

    def _leftoff_entry(self, overview: dict, cwd: str) -> Entry | None:
        next_steps = self._substantive_next_steps(overview)
        if not next_steps:
            return None

        headline = self._fallback_headline(overview)
        summary = f"Left off: {headline}"[:100]

        lines = [
            f"project: {overview.get('project') or '(unknown)'}",
            f"cwd: {cwd}",
        ]
        for label, key in (
            ("request", "request"),
            ("completed", "completed"),
            ("next_steps", "next_steps"),
            ("files_edited", "files_edited"),
            ("learned", "learned"),
        ):
            v = (overview.get(key) or "").strip()
            if v:
                # Cap learned in body to avoid re-bloat
                if key == "learned":
                    v = v[:1500]
                lines.append(f"{label}: {v}")

        body_parts = [f"## Trigger\n\n{TRIGGER_LEFTOFF}"]
        user_prompt = (overview.get("user_prompt") or "").strip()
        if user_prompt:
            body_parts.append("## User\n\n" + user_prompt[:2000])
        body_parts.append("## Left off\n\n" + "\n".join(lines))
        body = "\n\n".join(body_parts)
        return Entry(
            runtime=self.name,
            cwd=cwd,
            summary=summary,
            body=body,
            session_id=overview["memory_session_id"],
            project=overview.get("project") or "",
        )

    @staticmethod
    def _render_body(gate: dict, overview: dict) -> str:
        trigger = gate.get("trigger") or "lesson"
        lesson = gate.get("lesson_body") or "(no lesson body)"
        sections = [
            f"## Trigger\n\n{trigger}",
            f"## Lesson\n\n{lesson}",
        ]
        user_prompt = (overview.get("user_prompt") or "").strip()
        if user_prompt:
            sections.append("## User\n\n" + user_prompt[:2000])
        sess_lines = []
        for label, key in (
            ("request", "request"),
            ("learned", "learned"),
            ("next_steps", "next_steps"),
        ):
            v = (overview.get(key) or "").strip()
            if v:
                if key == "learned":
                    v = v[:1500]
                sess_lines.append(f"{label}: {v}")
        if sess_lines:
            sections.append("## Session\n\n" + "\n".join(sess_lines))
        return "\n\n".join(sections)
