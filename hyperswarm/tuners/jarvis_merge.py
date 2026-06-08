"""Cross-node Jarvis corpus merger.

Jarvis is Shawn-personal — one agent that runs on every node (Mac + NEB +
Cliqk + TRC). To train ONE Jarvis adapter from all interaction history,
this module:

1. rsyncs each remote source's session JSONLs into a local staging area
   (no-op for the local Mac source).
2. Walks all staged JSONLs, namespaces session IDs by host so cursors don't
   collide across hosts that happen to share a session UUID, and runs the
   same pair-extraction the per-server collector uses.

Output: one corpus.jsonl at ~/.openclaw/tune/<agent>/corpus.jsonl, ready
for `hyperswarm tune-train-local --agent <agent>`. Defaults to agent name
"jarvis" but the merger is agent-agnostic — re-using it for any future
cross-node persona only requires changing the default source paths.

State file (`~/.local/state/hyperswarm/tune/<agent>/jarvis-merge-cursors.json`)
is keyed by `host:session_id` so re-runs are idempotent and partial pulls
don't lose progress. Per-host collection state from `tune-collect` lives in
a separate file (`corpus-cursors.json`) — they don't conflict.

Per-host clawdbots stay company-isolated by design. They each run
`tune-collect` against their own server's sessions only. Only Jarvis fans
in across all nodes.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from hyperswarm.tuners.openclaw_corpus import (
    DEFAULT_CORPUS_BASE,
    DEFAULT_MAX_TURN_CHARS,
    DEFAULT_MIN_ASSISTANT_CHARS,
    DEFAULT_MIN_USER_CHARS,
    DEFAULT_STATE_DIR,
    _build_system_prompt,
    _filter_pair,
    _iter_pairs_from_lines,
    pair_to_example,
)

DEFAULT_STAGE_BASE = "~/.openclaw/tune/_jarvis-stage"


def _expand(p: str | os.PathLike) -> Path:
    return Path(os.path.expanduser(str(p)))


@dataclass
class CorpusSource:
    """One source of session JSONLs.

    `ssh_alias=None` means local — read directly from `remote_path`.
    Otherwise rsync `<ssh_alias>:<remote_path>/` into the stage dir.
    `host` is the label used to namespace cursors and tag examples; it must
    be unique across sources in a single merger run.
    """
    host: str
    remote_path: str
    ssh_alias: str | None = None  # None = local

    @property
    def is_local(self) -> bool:
        return self.ssh_alias is None


def default_sources() -> list[CorpusSource]:
    """Default Jarvis corpus sources: Mac + neb-server + cliqk-server + trc-server.

    Mac reads Claude Code session jsonls. The three servers read openclaw's
    Jarvis agent session jsonls. Each is overridable via the merger
    constructor or the CLI's --source flag.
    """
    return [
        CorpusSource(
            host="mac",
            remote_path=str(_expand("~/.claude/projects/-Users-screddy-projects")),
            ssh_alias=None,
        ),
        CorpusSource(
            host="neb-server",
            remote_path="~/.openclaw/agents/jarvis/sessions",
            ssh_alias="neb-server",
        ),
        CorpusSource(
            host="cliqk-server",
            remote_path="~/.openclaw/agents/jarvis/sessions",
            ssh_alias="cliqk-server",
        ),
        CorpusSource(
            host="trc-server",
            remote_path="~/.openclaw/agents/jarvis/sessions",
            ssh_alias="trc-server",
        ),
    ]


def _real_rsync(*, src: str, dst: Path, timeout: int = 600) -> tuple[int, str]:
    """Default rsync runner. -a preserves mtimes (needed for stable cursor
    behavior); --delete is *not* set — we want to keep files even if they're
    pruned upstream so partial corpus history isn't lost."""
    dst.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["rsync", "-a", "--include", "*.jsonl", "--include", "*/",
         "--exclude", "*", src, str(dst) + "/"],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return r.returncode, (r.stderr or r.stdout)


@dataclass
class JarvisCorpusMerger:
    agent: str = "jarvis"
    sources: list[CorpusSource] = field(default_factory=default_sources)
    stage_base: Path | None = None
    corpus_base: Path | None = None
    state_dir: Path | None = None
    system_prompt: str | None = None
    min_user_chars: int = DEFAULT_MIN_USER_CHARS
    min_assistant_chars: int = DEFAULT_MIN_ASSISTANT_CHARS
    max_chars: int = DEFAULT_MAX_TURN_CHARS

    # Test/inject hook: signature `(*, src: str, dst: Path) -> (rc, msg)`
    rsync_runner: Callable[..., tuple[int, str]] | None = None

    def __post_init__(self) -> None:
        self.stage_base = _expand(self.stage_base or DEFAULT_STAGE_BASE)
        self.corpus_base = _expand(self.corpus_base or DEFAULT_CORPUS_BASE)
        self.state_dir = _expand(self.state_dir or DEFAULT_STATE_DIR)
        seen = set()
        for s in self.sources:
            if s.host in seen:
                raise ValueError(f"duplicate host {s.host!r} in sources — must be unique")
            seen.add(s.host)

    # ── paths ───────────────────────────────────────────────────────────

    @property
    def corpus_path(self) -> Path:
        return self.corpus_base / self.agent / "corpus.jsonl"

    @property
    def state_path(self) -> Path:
        return self.state_dir / self.agent / "jarvis-merge-cursors.json"

    def _stage_dir_for(self, source: CorpusSource) -> Path:
        return self.stage_base / source.host

    # ── state ───────────────────────────────────────────────────────────

    def _load_state(self) -> dict:
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text())
            except (ValueError, json.JSONDecodeError):
                pass
        return {"cursors": {}, "examples_written": 0, "pulls": {}}

    def _save_state(self, state: dict) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2))

    # ── pull ────────────────────────────────────────────────────────────

    def pull_remotes(self) -> dict:
        """rsync each remote source into its stage dir. Local sources are
        no-ops — they're read directly from `remote_path` during collect.

        Returns per-host status. A failure on one host doesn't abort the
        others; the caller decides whether to proceed.
        """
        runner = self.rsync_runner or _real_rsync
        results = {}
        for source in self.sources:
            if source.is_local:
                results[source.host] = {"status": "local", "path": source.remote_path}
                continue
            stage = self._stage_dir_for(source)
            src = f"{source.ssh_alias}:{source.remote_path.rstrip('/')}/"
            try:
                rc, msg = runner(src=src, dst=stage)
            except Exception as e:
                results[source.host] = {"status": "failed", "reason": f"runner crashed: {e}"}
                continue
            results[source.host] = (
                {"status": "pulled", "stage": str(stage)}
                if rc == 0
                else {"status": "failed", "rc": rc, "reason": msg.strip()[:400]}
            )
        return results

    # ── collect ─────────────────────────────────────────────────────────

    def _read_dir(self, source: CorpusSource) -> Path:
        """Where to read JSONLs from for this source. Local: source.remote_path
        directly. Remote: the staged copy."""
        if source.is_local:
            return _expand(source.remote_path)
        return self._stage_dir_for(source)

    def collect(self) -> dict:
        """Walk all sources' JSONLs, append new pairs to corpus.jsonl, update
        per-(host, session_id) cursors. Idempotent — re-running picks up
        only newly-appended bytes per session."""
        state = self._load_state()
        cursors = state.setdefault("cursors", {})
        appended = 0
        sessions_seen = 0
        per_host = {s.host: {"sessions": 0, "appended": 0} for s in self.sources}

        self.corpus_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.corpus_path, "a") as out:
            for source in self.sources:
                read_dir = self._read_dir(source)
                if not read_dir.exists():
                    per_host[source.host]["status"] = "no-dir"
                    continue
                for jsonl in sorted(read_dir.glob("*.jsonl")):
                    if jsonl.name.endswith(".trajectory.jsonl"):
                        continue
                    sid = jsonl.stem
                    cursor_key = f"{source.host}:{sid}"
                    offset = int(cursors.get(cursor_key, 0))
                    with open(jsonl, "rb") as f:
                        f.seek(offset)
                        raw = f.read()
                        new_offset = offset + len(raw)
                    if not raw:
                        continue
                    sessions_seen += 1
                    per_host[source.host]["sessions"] += 1
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
                            system_prompt=self.system_prompt or _build_system_prompt(self.agent),
                        )
                        out.write(json.dumps(example) + "\n")
                        appended += 1
                        per_host[source.host]["appended"] += 1
                    cursors[cursor_key] = new_offset

        state["examples_written"] = state.get("examples_written", 0) + appended
        self._save_state(state)
        return {
            "agent": self.agent,
            "sessions_seen": sessions_seen,
            "appended": appended,
            "total_examples": state["examples_written"],
            "corpus_path": str(self.corpus_path),
            "per_host": per_host,
        }

    def run(self) -> dict:
        """Convenience: pull then collect."""
        pull = self.pull_remotes()
        collect = self.collect()
        return {"pull": pull, "collect": collect}


def merge_jarvis_corpus(**kwargs) -> dict:
    """Module-level convenience: pull + collect in one call."""
    return JarvisCorpusMerger(**kwargs).run()
