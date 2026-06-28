"""ClaudeMemSessionSource — session-level, significance-gated distillation.

This Source is the ONLY claude-mem→corpus path. It does NOT capture per-turn
raw transcript turns (the disabled `claude_code` / `codex` sources used to do
that and re-bloated the store to 26k). Instead it reads claude-mem's *session
overview* (`session_summaries`) at SESSION END and emits AT MOST ONE distilled
Entry — and only when the session contained a significant signal:

    (a) a CHANGE OF COURSE        — the plan/approach pivoted mid-session
    (b) a MISSED NORTH-STAR / PR  — the stated objective/PR was not achieved
    (c) a LESSON / TEACHING       — a generalizable correction, gotcha, or rule

LEFTOFF FALLBACK: when the LLM gate declines (or errors) but the session ended
with SUBSTANTIVE unfinished work — a non-boilerplate `next_steps` — capture()
emits ONE deterministic "leftoff" handoff Entry instead of None. This is the
"where Claude Code left off" state that Hermes' hyperswarm_search / prefetch
promises ("what was last left off on"): project, cwd, request, completed,
next_steps, files_edited, all in the body so keyword recall can match it.
Sessions with no next steps (or boilerplate like "No active work in
progress.") still distil to NOTHING — the store does not flood.

If neither a lesson nor a leftoff holds, capture() returns None and the CLI
writes nothing (cmd_capture treats `entry is None` as success). So a routine
session produces zero entries; only the significant ones reach the Supabase
training corpus.

Wiring:
  1. install() merges a `SessionEnd` hook into ~/.claude/settings.json that
     pipes the hook's stdin JSON to `hyperswarm capture --runtime
     claude_mem_session || true`. SessionEnd fires ONCE at teardown (unlike
     Stop, which fires every turn) — exactly the session-level cadence we want.
     If a host's Claude Code lacks SessionEnd, the install can also register a
     Stop hook; capture() self-guards by only distilling sessions claude-mem
     has marked status='completed', so even a per-turn Stop hook distills at
     most once per session.

  2. The SessionEnd hook stdin JSON is Claude Code's session-end payload:
        { "session_id": "<claude content session id>",
          "cwd": "/Users/.../projects/...", "hook_event_name": "SessionEnd" }

  3. capture(raw) opens ~/.claude-mem/claude-mem.db READ-ONLY (mode=ro,
     PRAGMA query_only=1 — claude-mem is NEVER modified), resolves the Claude
     `session_id` (== sdk_sessions.content_session_id) to claude-mem's
     `memory_session_id`, loads the session_summaries row(s), runs the gate,
     and returns ONE distilled Entry or None.

The distilled Entry then flows the EXISTING pipe (Scope.tag → MarkdownStore
→ rsync → shawn-corpus ingestion → Supabase). No corpus-side change is needed
for the qualifying path; the runtime is "claude-mem-session", which the
ingestion denylist intentionally does NOT exclude (it only excludes the
per-interaction "claude-mem-recall" runtime).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
from pathlib import Path

from hyperswarm.core.entry import Entry
from hyperswarm.core.source import Source

DEFAULT_SETTINGS_PATH = "~/.claude/settings.json"
DEFAULT_DB_PATH = "~/.claude-mem/claude-mem.db"
HOOK_MATCHER = ".*"
HOOK_SUBSTRING = "capture --runtime claude_mem_session"

# Trigger labels surfaced by the gate.
TRIGGER_COURSE_CHANGE = "course_change"
TRIGGER_MISSED_PR = "missed_pr"
TRIGGER_LESSON = "lesson"
_VALID_TRIGGERS = {TRIGGER_COURSE_CHANGE, TRIGGER_MISSED_PR, TRIGGER_LESSON}

# Deterministic fallback trigger — NOT an LLM-gate trigger. Emitted when the
# gate declines/errors but the session left substantive unfinished work.
TRIGGER_LEFTOFF = "leftoff"

# claude-mem writes filler next_steps for idle/finished sessions; none of
# these constitute a handoff worth storing.
_BOILERPLATE_NEXT_STEPS = re.compile(
    r"^(none|n/?a|nothing|all done|done|complete[d]?)[.!]?$"
    r"|^no\s+(active|further|next|outstanding|remaining|pending|immediate)\b",
    re.IGNORECASE,
)


def _resolve_hyperswarm_binary() -> str:
    return shutil.which("hyperswarm") or "hyperswarm"


def _build_default_hook_command() -> str:
    binary = _resolve_hyperswarm_binary()
    return f"'{binary}' capture --runtime claude_mem_session || true"


DEFAULT_HOOK_COMMAND = "hyperswarm capture --runtime claude_mem_session || true"


# --------------------------------------------------------------------------- gate

_GATE_SYSTEM_PROMPT = """You are a significance gate for a long-term engineering memory corpus.

You are given a SESSION OVERVIEW (request, what was investigated, what was learned, what was completed, next steps, notes) distilled by claude-mem for ONE coding session.

Decide whether this session is worth retaining as ONE distilled lesson. It qualifies ONLY if it contained at least one of:

  (a) course_change — the plan or approach PIVOTED mid-session (an approach was tried and abandoned for another, a wrong assumption was corrected, a strategy changed).
  (b) missed_pr — the stated objective / north-star / PR was NOT achieved, was abandoned, blocked, or left incomplete.
  (c) lesson — a GENERALIZABLE correction, gotcha, constraint, or rule worth remembering next time (not a one-off restatement of the task).

It does NOT qualify if it is a routine success with nothing generalizable to teach: the request was completed straightforwardly, no pivots, no surprises, "learned"/"notes"/"next_steps" are empty or merely restate the task.

Return ONLY strict JSON — no preamble, no trailing prose:
{
  "qualifies": true | false,
  "trigger": "course_change" | "missed_pr" | "lesson" | null,
  "headline": "string, max 100 chars — one-line summary of the lesson/pivot/miss, or '' if not qualifying",
  "lesson_body": "string, max 800 chars — the durable lesson/teaching in plain prose, or '' if not qualifying",
  "reason": "string, max 200 chars — why it does or does not qualify"
}

Rules:
- If qualifies is false, trigger must be null and headline/lesson_body must be "".
- Be conservative. Missing a borderline session is fine; flooding the corpus with routine work is not.
"""


def _parse_gate_json(raw: str) -> dict:
    """Tolerantly parse codex/LLM output into a gate dict. Safe default = not qualifying."""
    not_qual = {"qualifies": False, "trigger": None, "headline": "", "lesson_body": "", "reason": "parse_default"}
    if not raw or not raw.strip():
        return not_qual

    data = None
    try:
        data = json.loads(raw.strip())
    except json.JSONDecodeError:
        cleaned = raw.strip()
        brace = cleaned.find("{")
        end = cleaned.rfind("}")
        if 0 <= brace < end:
            cleaned = cleaned[brace : end + 1]
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            try:
                import json_repair  # type: ignore

                repaired = json_repair.loads(cleaned)
                if isinstance(repaired, dict):
                    data = repaired
            except Exception:
                data = None
    if not isinstance(data, dict):
        return not_qual

    # Strict: only a real JSON boolean true (or int 1) qualifies. Do NOT use
    # bool(), which truthiness-coerces a stringized "false"/"no"/"0" to True and
    # would wrongly emit a corpus entry for a routine session. Default-deny.
    _q = data.get("qualifies")
    qualifies = _q is True or _q == 1
    trigger = data.get("trigger")
    if not qualifies or trigger not in _VALID_TRIGGERS:
        # Either non-qualifying or an invalid/missing trigger → treat as not qualifying.
        if qualifies and trigger not in _VALID_TRIGGERS:
            return {**not_qual, "reason": "invalid_trigger"}
        return not_qual
    return {
        "qualifies": True,
        "trigger": trigger,
        "headline": (data.get("headline") or "").strip()[:100],
        "lesson_body": (data.get("lesson_body") or "").strip()[:800],
        "reason": (data.get("reason") or "").strip()[:200],
    }


class ClaudeMemSessionSource(Source):
    name = "claude-mem-session"

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self.settings_path = Path(
            os.path.expanduser(self.config.get("settings_path", DEFAULT_SETTINGS_PATH))
        )
        self.db_path = Path(os.path.expanduser(self.config.get("db_path", DEFAULT_DB_PATH)))
        self.hook_command = self.config.get("hook_command") or _build_default_hook_command()
        # Which Claude Code hook event to install on. SessionEnd is preferred
        # (fires once at teardown). Fallback: "Stop" (fires every turn, but
        # capture() self-guards to distill at most once per completed session).
        self.hook_event = self.config.get("hook_event", "SessionEnd")
        # Codex gate config (reuses the codex CLI pattern; ~12s, free OAuth).
        self.gate_model = self.config.get("gate_model", "gpt-5.4")
        self.gate_timeout = int(self.config.get("gate_timeout", 60))
        # Test seam: inject a callable(prompt)->str to avoid shelling out to codex.
        self._gate_fn = self.config.get("gate_fn")

    # ------------------------------------------------------------- install
    def install(self) -> None:
        """Idempotent: re-running does not duplicate the hook."""
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
                    if not (h.get("type") == "command" and HOOK_SUBSTRING in h.get("command", ""))
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
        """raw is the SessionEnd (or Stop) hook stdin JSON.

        Returns ONE distilled Entry if the session qualifies, else None.
        Never raises — a hook firing this must not crash the user's session.
        """
        cwd = raw.get("cwd", "") or os.getcwd()
        content_session_id = raw.get("session_id", "") or ""
        if not content_session_id:
            return None
        if not self.db_path.exists():
            return None

        try:
            overview = self._read_session_overview(content_session_id)
        except Exception:
            # Read failure must never crash the user's session.
            return None
        if overview is None:
            return None

        # If this came from a per-turn Stop hook, only distill once the session
        # is marked completed by claude-mem (so we don't distill mid-session
        # every turn). SessionEnd hooks don't need this guard but it's harmless.
        if self.hook_event != "SessionEnd" and overview.get("status") != "completed":
            return None

        # --- Tier 1: cheap structural pre-filter (no LLM) --------------------
        if not self._structurally_significant(overview):
            return None

        # --- Tier 2: LLM significance gate -----------------------------------
        gate = self._run_gate(overview)
        if not gate.get("qualifies"):
            # LEFTOFF FALLBACK — deterministic, no LLM. A session that is not
            # lesson-worthy but ended with substantive unfinished work still
            # gets ONE handoff entry so Hermes can pick up where Claude Code
            # left off. Also covers gate errors (codex down): the handoff
            # state must not depend on the LLM being reachable.
            return self._leftoff_entry(overview, cwd)

        memory_session_id = overview["memory_session_id"]
        summary = gate.get("headline") or self._fallback_headline(overview)
        body = self._render_body(gate, overview)

        return Entry(
            runtime=self.name,
            # Keep the REAL cwd so the orchestrator's git_remote Scope plugin can
            # resolve the org/company tag. claude-mem's `project` is a short slug
            # (e.g. "projects"), not a path, so it must NOT replace cwd; it rides
            # along on the Entry.project field instead.
            cwd=cwd,
            summary=summary,
            body=body,
            session_id=memory_session_id,  # STABLE — drives idempotency downstream
            project=overview.get("project") or "",
        )

    # ------------------------------------------------------------- db read
    def _connect_ro(self) -> sqlite3.Connection:
        """Open claude-mem.db strictly read-only (WAL-aware). NEVER mutates."""
        uri = f"file:{self.db_path}?mode=ro&immutable=0"
        conn = sqlite3.connect(uri, uri=True, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only = 1")
        return conn

    def _read_session_overview(self, content_session_id: str) -> dict | None:
        """Resolve the Claude content_session_id → claude-mem session overview.

        Returns a dict with the session_summaries fields + project + status +
        memory_session_id, or None if the session isn't found / has no summary.
        """
        conn = self._connect_ro()
        try:
            sdk = conn.execute(
                "SELECT content_session_id, memory_session_id, project, user_prompt, status "
                "FROM sdk_sessions WHERE content_session_id = ?",
                (content_session_id,),
            ).fetchone()
            if sdk is None:
                return None
            memory_session_id = sdk["memory_session_id"]
            if not memory_session_id:
                # Session never completed → claude-mem hasn't minted a memory id.
                return None

            rows = conn.execute(
                "SELECT request, investigated, learned, completed, next_steps, "
                "       files_read, files_edited, notes "
                "FROM session_summaries WHERE memory_session_id = ? "
                "ORDER BY prompt_number DESC, created_at_epoch DESC",
                (memory_session_id,),
            ).fetchall()
            if not rows:
                return None

            # Collapse multiple summary rows (multi-prompt session) into one
            # overview, preferring the latest non-empty value per field.
            def pick(field: str) -> str:
                for r in rows:
                    v = (r[field] or "").strip() if r[field] is not None else ""
                    if v:
                        return v
                return ""

            obs_count = conn.execute(
                "SELECT COUNT(*) AS c FROM observations WHERE memory_session_id = ?",
                (memory_session_id,),
            ).fetchone()["c"]

            return {
                "memory_session_id": memory_session_id,
                "content_session_id": content_session_id,
                "project": sdk["project"] or "",
                "user_prompt": sdk["user_prompt"] or "",
                "status": sdk["status"] or "",
                "request": pick("request"),
                "investigated": pick("investigated"),
                "learned": pick("learned"),
                "completed": pick("completed"),
                "next_steps": pick("next_steps"),
                "files_read": pick("files_read"),
                "files_edited": pick("files_edited"),
                "notes": pick("notes"),
                "observation_count": obs_count,
            }
        finally:
            conn.close()

    # ------------------------------------------------------------- gating
    @staticmethod
    def _structurally_significant(overview: dict) -> bool:
        """Cheap pre-filter: drop sessions with nothing to teach before the LLM.

        A session is a candidate only if it has non-trivial learned / notes /
        next_steps content. Routine successes (completed with empty teaching
        fields) are dropped here, saving the LLM call.
        """
        teaching = " ".join(
            (overview.get(k) or "") for k in ("learned", "notes", "next_steps")
        ).strip()
        # Require some minimal substance — a couple of words, not a stray token.
        return len(teaching) >= 12

    def _run_gate(self, overview: dict) -> dict:
        """Run the significance gate (codex CLI by default). Safe default = not qualifying."""
        prompt = _GATE_SYSTEM_PROMPT + "\n\nSESSION OVERVIEW:\n" + self._overview_text(overview)
        try:
            if self._gate_fn is not None:
                raw = self._gate_fn(prompt)
            else:
                raw = self._call_codex(prompt)
        except Exception:
            return {"qualifies": False, "trigger": None, "headline": "", "lesson_body": "", "reason": "gate_error"}
        return _parse_gate_json(raw)

    def _call_codex(self, prompt: str) -> str:
        # Reuse shawn-corpus's codex subprocess wrapper if available (same
        # pattern as CodexPivotClassifier); fall back to a local invocation.
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
                [codex_bin, "exec", "--model", self.gate_model, "--color", "never",
                 "--skip-git-repo-check", "--output-last-message", str(out_path)],
                input=prompt, capture_output=True, text=True, timeout=self.gate_timeout,
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
        text = overview.get("request") or overview.get("learned") or "(claude-mem session)"
        return text.replace("\n", " ").strip()[:100]

    # ------------------------------------------------------------- leftoff
    @staticmethod
    def _substantive_next_steps(overview: dict) -> str:
        """Return the next_steps text iff it describes real unfinished work.

        Empty, too-short, or boilerplate values ("No active work in
        progress.", "None", "Done") return "" — no handoff to record.
        """
        text = (overview.get("next_steps") or "").strip()
        if len(text) < 12:
            return ""
        if _BOILERPLATE_NEXT_STEPS.match(text):
            return ""
        return text

    def _leftoff_entry(self, overview: dict, cwd: str) -> Entry | None:
        """Build the deterministic handoff Entry, or None if nothing to hand off."""
        next_steps = self._substantive_next_steps(overview)
        if not next_steps:
            return None

        headline = self._fallback_headline(overview)
        summary = f"Left off: {headline}"[:100]

        # Everything Hermes needs to resume rides in the BODY: its prefetch /
        # hyperswarm_search score keyword overlap against entry.body only, so
        # the project name, cwd, and file paths must appear here verbatim.
        lines = [
            f"project: {overview.get('project') or '(unknown)'}",
            f"cwd: {cwd}",
        ]
        for label, key in (
            ("request", "request"),
            ("completed", "completed"),
            ("next_steps", "next_steps"),
            ("files_edited", "files_edited"),
        ):
            v = (overview.get(key) or "").strip()
            if v:
                lines.append(f"{label}: {v}")

        body_parts = [f"## Trigger\n\n{TRIGGER_LEFTOFF}"]
        # Carry Shawn's verbatim prompt here too (same rationale as _render_body).
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
            session_id=overview["memory_session_id"],  # same idempotency key
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
        # Emit Shawn's VERBATIM first prompt under a `## User` header. This
        # preserves his real words (better training signal than the paraphrased
        # request:/learned: summary) AND satisfies the corpus learning prefilter,
        # which keys on a Shawn-turn marker — without it every distilled session
        # was skipped 'no_shawn_turns' and the learnings table went flat
        # (capture-format mismatch, 2026-06-27). Cap length to avoid re-bloating.
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
                sess_lines.append(f"{label}: {v}")
        if sess_lines:
            sections.append("## Session\n\n" + "\n".join(sess_lines))
        return "\n\n".join(sections)
