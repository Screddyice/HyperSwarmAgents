"""OpenClaw fine-tune corpus collector.

Reads session JSONLs (same source as the reflector), pairs consecutive
user→assistant turns, and appends them to a JSONL fine-tune corpus in OpenAI's
chat-format:

    {"messages": [
        {"role": "system", "content": "<persona prompt>"},
        {"role": "user", "content": "<shawn's prompt>"},
        {"role": "assistant", "content": "<agent's response>"}
    ]}

One example per user→assistant pair. The system prompt is a short personalized
preamble naming the agent and the user (Shawn) so the fine-tune learns
"this is jarvis-for-Shawn" rather than generic agent behavior.

Corpus state lives at ~/.local/state/hyperswarm/tune/<agent>/corpus-cursors.json
The corpus file itself lives at ~/.openclaw/tune/<agent>/corpus.jsonl by default.
Cursors track per-session byte offset so re-runs are idempotent.

Quality filters applied before writing an example:
- Skip if user message <= 10 chars (likely a typo or empty)
- Skip if assistant message <= 20 chars (likely a non-substantive response)
- Skip if either side > 8000 chars (oversized; trim earlier in the pipeline)
- Skip if last assistant message in session has not been "stable" for N seconds
  (caller's responsibility — this module does not look at mtime)
"""
from __future__ import annotations

import json
import os
import socket
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

DEFAULT_AGENTS_DIR = "~/.openclaw/agents"
DEFAULT_CORPUS_BASE = "~/.openclaw/tune"
DEFAULT_STATE_DIR = "~/.local/state/hyperswarm/tune"
DEFAULT_MIN_USER_CHARS = 10
DEFAULT_MIN_ASSISTANT_CHARS = 20
DEFAULT_MAX_TURN_CHARS = 8000


def _expand(p: str | os.PathLike) -> Path:
    return Path(os.path.expanduser(str(p)))


def _build_system_prompt(agent: str) -> str:
    """Default system prompt for fine-tune examples.

    Keep it short — the fine-tune learns from the assistant content, not from
    a verbose system role. The point of the system prompt here is to anchor
    the model on "personalized for Shawn" rather than generic chat.
    """
    return (
        f"You are {agent}, Shawn Reddy's personal AI assistant. "
        "Respond in the style and with the preferences he has demonstrated "
        "across this conversation history."
    )


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict):
                t = c.get("text")
                if t:
                    parts.append(t)
            else:
                parts.append(str(c))
        return " ".join(parts).strip()
    return ""


@dataclass
class TurnPair:
    user: str
    assistant: str
    timestamp: str  # of the assistant turn


def _iter_pairs_from_lines(lines: Iterator[str]) -> Iterator[TurnPair]:
    """Pair consecutive user→assistant messages in a session jsonl.

    Tool result and other event types are skipped. If two user messages
    appear back-to-back the older one is dropped (we keep the most recent
    user message before each assistant message).
    """
    pending_user: str | None = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except (ValueError, json.JSONDecodeError):
            continue
        if d.get("type") != "message":
            continue
        msg = d.get("message") or {}
        role = msg.get("role")
        text = _extract_text(msg.get("content"))
        if not text:
            continue
        if role == "user":
            pending_user = text
        elif role == "assistant" and pending_user is not None:
            yield TurnPair(
                user=pending_user,
                assistant=text,
                timestamp=d.get("timestamp", ""),
            )
            pending_user = None
        # other roles (toolResult etc.) ignored


def _filter_pair(
    pair: TurnPair,
    *,
    min_user_chars: int,
    min_assistant_chars: int,
    max_chars: int,
) -> bool:
    if len(pair.user) < min_user_chars:
        return False
    if len(pair.assistant) < min_assistant_chars:
        return False
    if len(pair.user) > max_chars or len(pair.assistant) > max_chars:
        return False
    return True


def pair_to_example(
    pair: TurnPair,
    *,
    agent: str,
    system_prompt: str | None = None,
) -> dict:
    """Convert one TurnPair into an OpenAI fine-tune chat example."""
    return {
        "messages": [
            {"role": "system", "content": system_prompt or _build_system_prompt(agent)},
            {"role": "user", "content": pair.user},
            {"role": "assistant", "content": pair.assistant},
        ]
    }


@dataclass
class OpenClawCorpusCollector:
    agent: str
    host: str = ""
    agents_dir: Path | None = None
    corpus_base: Path | None = None
    state_dir: Path | None = None
    system_prompt: str | None = None
    min_user_chars: int = DEFAULT_MIN_USER_CHARS
    min_assistant_chars: int = DEFAULT_MIN_ASSISTANT_CHARS
    max_chars: int = DEFAULT_MAX_TURN_CHARS

    def __post_init__(self) -> None:
        self.host = self.host or socket.gethostname()
        self.agents_dir = _expand(self.agents_dir or DEFAULT_AGENTS_DIR)
        self.corpus_base = _expand(self.corpus_base or DEFAULT_CORPUS_BASE)
        self.state_dir = _expand(self.state_dir or DEFAULT_STATE_DIR)

    @property
    def sessions_dir(self) -> Path:
        return self.agents_dir / self.agent / "sessions"

    @property
    def corpus_path(self) -> Path:
        return self.corpus_base / self.agent / "corpus.jsonl"

    @property
    def state_path(self) -> Path:
        return self.state_dir / self.agent / "corpus-cursors.json"

    def _load_state(self) -> dict:
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text())
            except (ValueError, json.JSONDecodeError):
                return {"cursors": {}, "examples_written": 0}
        return {"cursors": {}, "examples_written": 0}

    def _save_state(self, state: dict) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2))

    def run(self) -> dict:
        if not self.sessions_dir.exists():
            return {"agent": self.agent, "status": "no-sessions-dir", "appended": 0}
        state = self._load_state()
        cursors = state.setdefault("cursors", {})
        appended = 0
        sessions_seen = 0
        self.corpus_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.corpus_path, "a") as out:
            for jsonl in sorted(self.sessions_dir.glob("*.jsonl")):
                if jsonl.name.endswith(".trajectory.jsonl"):
                    continue
                sid = jsonl.stem
                offset = int(cursors.get(sid, 0))
                with open(jsonl, "rb") as f:
                    f.seek(offset)
                    raw = f.read()
                    new_offset = offset + len(raw)
                if not raw:
                    continue
                sessions_seen += 1
                lines = raw.decode("utf-8", errors="replace").splitlines()
                for pair in _iter_pairs_from_lines(iter(lines)):
                    if not _filter_pair(
                        pair,
                        min_user_chars=self.min_user_chars,
                        min_assistant_chars=self.min_assistant_chars,
                        max_chars=self.max_chars,
                    ):
                        continue
                    example = pair_to_example(
                        pair,
                        agent=self.agent,
                        system_prompt=self.system_prompt,
                    )
                    out.write(json.dumps(example) + "\n")
                    appended += 1
                cursors[sid] = new_offset
        state["examples_written"] = state.get("examples_written", 0) + appended
        self._save_state(state)
        return {
            "agent": self.agent,
            "host": self.host,
            "sessions_seen": sessions_seen,
            "appended": appended,
            "total_examples": state["examples_written"],
            "corpus_path": str(self.corpus_path),
        }


def collect_corpus(agent: str, **kwargs) -> dict:
    """Module-level convenience: build collector, run one pass."""
    return OpenClawCorpusCollector(agent=agent, **kwargs).run()
