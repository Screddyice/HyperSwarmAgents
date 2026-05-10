"""OpenClaw session reflector — distill jarvis/clawdbot session JSONLs into
curated memory files.

Architecture:
  1. read_new_turns(): for each session JSONL under
     ~/.openclaw/agents/<agent>/sessions/, read user/assistant turns past the
     stored cursor.
  2. reflect(): for each session with new turns, call an LLM with the
     reflection prompt. Output is zero-or-more YAML-frontmatter markdown
     blocks, separated by `---` lines.
  3. write_memories(): split LLM output into individual blocks and write
     each as `<date>_<session_short>_<content_hash>.md` into
     server-learned/<agent>/. Hashes prevent duplicate writes across runs.
  4. save_cursor(): persist per-session offset so the next run picks up
     where this one left off.

Cursor state lives at ~/.local/state/hyperswarm/reflect/<agent>.json
The output dir defaults to the openclaw memory_search extraPath that
syncs back to Mac auto-memory:
  ~/.openclaw/claude-code-history/projects/-Users-screddy-projects/memory/server-learned/<agent>/

LLM call shape: OpenAI Chat Completions, model configurable via env
HYPERSWARM_REFLECT_MODEL (default: gpt-4o-mini for cost). Auth via
OPENAI_API_KEY in environment. See repo README for cost notes.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DEFAULT_AGENTS_DIR = "~/.openclaw/agents"
DEFAULT_OUTPUT_BASE = (
    "~/.openclaw/claude-code-history/projects/-Users-screddy-projects/memory/server-learned"
)
DEFAULT_STATE_DIR = "~/.local/state/hyperswarm/reflect"
DEFAULT_MODEL = os.environ.get("HYPERSWARM_REFLECT_MODEL", "gpt-4o-mini")
DEFAULT_MAX_TURNS = 60
DEFAULT_MAX_TURN_CHARS = 800

REFLECTION_SYSTEM_PROMPT = """You are reading raw session turns from {agent} on {host}, an AI assistant for Shawn Reddy.
Your job: extract HIGH-SIGNAL learnings — facts future-{agent} or other instances should know.

OUTPUT ONLY new memories worth keeping. Skip:
- Conversation transcripts or session summaries (the raw is already stored).
- Things obvious from the current code, repos, or environment state.
- One-off task details, ephemeral state, or in-flight work.
- Anything already documented in known memory.
- **Recurring scheduled-workflow descriptions.** If the session is a cron/scheduled
  agent run executing the same workflow that's run dozens of times before
  (meeting prep, daily digest, post-meeting recap, status sweep, health check,
  etc.), DO NOT emit a memory describing the workflow. The workflow itself
  lives in the skill/code; it's not a learning. Only emit a memory if Shawn
  said something NEW about the workflow during this session (e.g. "from now
  on, also include X in the brief" — that's feedback worth keeping).
- **Anything that would be the same answer if asked again next week.** A memory
  that captures "how to run the meeting prep workflow" doesn't get smarter
  with repetition; it just clogs the index. Recurring task definitions belong
  in code or docs, not in distilled-memory form.

Memory types and when to use each:
- user: facts about Shawn's role, preferences, knowledge. Tailor future behavior.
- feedback: corrections he gave you, OR non-obvious approaches he validated. Include WHY.
- project: company/initiative state changes — names, deadlines, decisions.
- reference: external system pointers (Linear projects, Slack channels, dashboards).

Output format (YAML frontmatter, separated by `---` lines, ZERO blocks if nothing rises to the bar):

---
name: short title
description: one-line relevance hook for future search
type: user
---
body line 1
body line 2

---
name: ...
description: ...
type: feedback
---
Rule itself.

**Why:** the reason it was given (cite the incident if any).

**How to apply:** when this rule kicks in.

If nothing high-signal exists in these turns, output literally NOTHING (empty response). Do not invent."""


@dataclass(frozen=True)
class TurnExcerpt:
    """A trimmed user/assistant turn ready for the prompt."""

    role: str  # "user" | "assistant"
    text: str
    timestamp: str


def _expand(p: str | os.PathLike) -> Path:
    return Path(os.path.expanduser(str(p)))


def extract_turn(line: str, *, max_chars: int = DEFAULT_MAX_TURN_CHARS) -> TurnExcerpt | None:
    """Pull a user/assistant message out of one JSONL line, or None."""
    try:
        d = json.loads(line)
    except (ValueError, json.JSONDecodeError):
        return None
    if d.get("type") != "message":
        return None
    msg = d.get("message") or {}
    role = msg.get("role")
    if role not in ("user", "assistant"):
        return None
    content = msg.get("content")
    if isinstance(content, list):
        text = " ".join(
            (c.get("text") or "") if isinstance(c, dict) else str(c) for c in content
        )
    elif isinstance(content, str):
        text = content
    else:
        text = ""
    text = text.strip()
    if not text:
        return None
    if len(text) > max_chars:
        text = text[:max_chars] + "…"
    return TurnExcerpt(role=role, text=text, timestamp=d.get("timestamp", ""))


def read_new_turns(
    session_path: Path,
    *,
    cursor_offset: int = 0,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> tuple[list[TurnExcerpt], int]:
    """Read past-the-cursor user/assistant turns from one session JSONL.

    Returns (turns, new_cursor_offset). `new_cursor_offset` is the byte
    position to store for the next run — set to file size after a clean read,
    so partial-line writes don't desync us.
    """
    turns: list[TurnExcerpt] = []
    if not session_path.exists():
        return turns, cursor_offset
    with open(session_path, "rb") as f:
        f.seek(cursor_offset)
        raw = f.read()
        new_offset = cursor_offset + len(raw)
    text = raw.decode("utf-8", errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        t = extract_turn(line)
        if t:
            turns.append(t)
    if max_turns and len(turns) > max_turns:
        turns = turns[-max_turns:]
    return turns, new_offset


def build_messages(
    *,
    agent: str,
    host: str,
    session_id: str,
    turns: Iterable[TurnExcerpt],
) -> list[dict]:
    """Build an OpenAI-style messages list for the reflection call."""
    excerpt_lines = []
    for t in turns:
        excerpt_lines.append(f"[{t.timestamp}] {t.role}: {t.text}")
    excerpt = "\n\n".join(excerpt_lines)
    system = REFLECTION_SYSTEM_PROMPT.format(agent=agent, host=host)
    user = (
        f"Session id: {session_id}\n"
        f"Below are recent {sum(1 for _ in turns) if False else len(excerpt_lines)} user/assistant turns. "
        f"Extract any high-signal memories per the schema. Respond with the markdown blocks only "
        f"(or nothing if no memory rises to the bar).\n\n---\n\n{excerpt}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


_BLOCK_SPLIT_RE = re.compile(r"^\s*---\s*$", re.MULTILINE)


def split_memory_blocks(llm_output: str) -> list[str]:
    """Split a multi-block LLM response into individual frontmatter+body blocks.

    Each block returned is well-formed: starts with `---`, has frontmatter,
    closes with `---`, then body. Blocks without complete frontmatter are
    discarded (defensive against half-formed LLM output).
    """
    text = (llm_output or "").strip()
    if not text:
        return []
    # Normalize: ensure leading `---` so the first block is recognized.
    if not text.startswith("---"):
        return []
    # Walk the doc; each block is from a leading `---` through the next
    # `---` that ends frontmatter, then through the next leading `---`
    # at the start of a new block (or EOF).
    blocks: list[str] = []
    pieces = re.split(r"^---\s*$", text, flags=re.MULTILINE)
    # pieces[0] is empty (leading ---), then alternating frontmatter / body /
    # frontmatter / body / ... — but our LLM emits `---\nfrontmatter\n---\nbody`,
    # so the pattern in split is: ['', frontmatter, body, frontmatter, body, ...]
    i = 1
    while i + 1 < len(pieces):
        front = pieces[i].strip()
        body_and_more = pieces[i + 1]
        if not front or "name:" not in front:
            # malformed; skip both
            i += 2
            continue
        # body runs until the next `---` start-of-block or EOF
        # if there's another `---` after, body ends before it (handled by next loop iter)
        block = f"---\n{front}\n---\n{body_and_more.strip()}\n"
        blocks.append(block)
        i += 2
    return blocks


def _semantic_hash(block: str) -> str:
    """Hash (name, description) only — for dedup across sessions/dates.

    Two memories with the same name+description should not BOTH live in the
    server-learned dir, even if they were generated weeks apart. We use a
    short stable hash so a glob over `*_<hash>.md` finds the one canonical
    file regardless of date prefix.
    """
    m = re.search(r"^name:\s*(.+?)\s*$", block, flags=re.MULTILINE)
    name = (m.group(1) if m else "").strip().lower()
    m = re.search(r"^description:\s*(.+?)\s*$", block, flags=re.MULTILINE)
    desc = (m.group(1) if m else "").strip().lower()
    key = f"{name}::{desc}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]


def write_memory_block(
    *,
    block: str,
    agent: str,
    host: str,
    session_id: str,
    timestamp: str,
    output_dir: Path,
) -> Path | None:
    """Write one memory block to disk, augmenting frontmatter with provenance.

    Dedup: filenames carry a hash of (name, description) only. If any existing
    file in output_dir ends in `_<hash>.md`, the new memory is treated as a
    duplicate and skipped. This prevents "meeting prep workflow" reflections
    from compounding session-after-session.

    Returns the written path, or None if a duplicate already exists.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # Inject provenance into frontmatter (idempotent — only if missing)
    augmented = _inject_provenance(
        block,
        agent=agent,
        host=host,
        session_id=session_id,
        timestamp=timestamp,
    )
    name_hash = _semantic_hash(augmented)
    # Skip if any existing file with this semantic hash is present (regardless
    # of date / session prefix). This is the dedup gate.
    for existing in output_dir.glob(f"*_{name_hash}.md"):
        return None
    date = datetime.date.today().isoformat()
    short = (session_id or "unknown")[:8]
    fpath = output_dir / f"{date}_{short}_{name_hash}.md"
    if fpath.exists():
        return None
    fpath.write_text(augmented, encoding="utf-8")
    return fpath


def _inject_provenance(
    block: str,
    *,
    agent: str,
    host: str,
    session_id: str,
    timestamp: str,
) -> str:
    """Add originAgent/originHost/originSession/originTimestamp to frontmatter
    if not already present, leaving the body and existing fields intact.
    """
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", block, flags=re.DOTALL)
    if not m:
        return block
    front, body = m.group(1), m.group(2)
    additions = []
    for key, val in (
        ("originAgent", agent),
        ("originHost", host),
        ("originSession", session_id),
        ("originTimestamp", timestamp),
    ):
        if not val:
            continue
        if not re.search(rf"^{key}:", front, flags=re.MULTILINE):
            additions.append(f"{key}: {val}")
    if additions:
        front = front.rstrip() + "\n" + "\n".join(additions)
    return f"---\n{front}\n---\n{body.rstrip()}\n"


@dataclass
class OpenClawSessionReflector:
    """Read-and-distill driver. Construct, then call .run()."""

    agent: str
    host: str = ""
    agents_dir: Path | None = None
    output_base: Path | None = None
    state_dir: Path | None = None
    model: str = DEFAULT_MODEL
    max_turns: int = DEFAULT_MAX_TURNS
    llm_call: callable | None = None  # injectable for tests

    def __post_init__(self) -> None:
        self.host = self.host or socket.gethostname()
        self.agents_dir = _expand(self.agents_dir or DEFAULT_AGENTS_DIR)
        self.output_base = _expand(self.output_base or DEFAULT_OUTPUT_BASE)
        self.state_dir = _expand(self.state_dir or DEFAULT_STATE_DIR)

    @property
    def sessions_dir(self) -> Path:
        return self.agents_dir / self.agent / "sessions"

    @property
    def output_dir(self) -> Path:
        return self.output_base / self.agent

    @property
    def state_path(self) -> Path:
        return self.state_dir / f"{self.agent}.json"

    def _load_state(self) -> dict:
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text())
            except (ValueError, json.JSONDecodeError):
                return {"cursors": {}}
        return {"cursors": {}}

    def _save_state(self, state: dict) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2))

    def _call_llm(self, messages: list[dict]) -> str:
        if self.llm_call is not None:
            return self.llm_call(messages)
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "openai package not installed; pip install openai"
            ) from e
        client = OpenAI()
        rsp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
        )
        return (rsp.choices[0].message.content or "").strip()

    def run(self) -> dict:
        """Single reflection pass. Returns a result summary dict."""
        if not self.sessions_dir.exists():
            return {"agent": self.agent, "status": "no-sessions-dir", "written": 0}
        state = self._load_state()
        cursors = state.setdefault("cursors", {})
        written: list[str] = []
        sessions_processed = 0
        for jsonl in sorted(self.sessions_dir.glob("*.jsonl")):
            # Skip trajectory files (different schema)
            if jsonl.name.endswith(".trajectory.jsonl"):
                continue
            sid = jsonl.stem
            old_offset = int(cursors.get(sid, 0))
            turns, new_offset = read_new_turns(
                jsonl, cursor_offset=old_offset, max_turns=self.max_turns
            )
            cursors[sid] = new_offset
            if not turns:
                continue
            sessions_processed += 1
            messages = build_messages(
                agent=self.agent, host=self.host, session_id=sid, turns=turns
            )
            try:
                output = self._call_llm(messages)
            except Exception as e:
                # Keep cursor; log and move on so a flaky LLM call doesn't
                # poison the whole run.
                print(f"reflect: llm call failed for {sid}: {e}", file=sys.stderr)
                continue
            blocks = split_memory_blocks(output)
            last_ts = turns[-1].timestamp if turns else ""
            for block in blocks:
                fpath = write_memory_block(
                    block=block,
                    agent=self.agent,
                    host=self.host,
                    session_id=sid,
                    timestamp=last_ts,
                    output_dir=self.output_dir,
                )
                if fpath:
                    written.append(str(fpath))
        self._save_state(state)
        return {
            "agent": self.agent,
            "host": self.host,
            "sessions_processed": sessions_processed,
            "written": len(written),
            "files": written,
        }


def reflect_agent(agent: str, **kwargs) -> dict:
    """Module-level convenience: build a Reflector and run one pass."""
    return OpenClawSessionReflector(agent=agent, **kwargs).run()
