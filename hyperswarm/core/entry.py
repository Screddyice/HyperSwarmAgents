"""Entry — the unit of captured context.

An Entry is the on-disk shape every Source produces and every Store persists.
Markdown body + frontmatter — see README for the canonical layout.
"""
from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass, field


@dataclass
class Entry:
    runtime: str                      # e.g. "claude-code", "codex", "openclaw-neb"
    cwd: str                          # absolute cwd at capture time
    summary: str                      # one-line digest used by `recent`
    body: str                         # markdown body (decisions, files, blockers)
    session_id: str = ""              # runtime-provided session id, or ""
    scope: str = ""                   # filled in by orchestrator via Scope plugin
    project: str = ""                 # optional project name (e.g. git repo name); Scope plugins may set
    timestamp: _dt.datetime = field(
        default_factory=lambda: _dt.datetime.now(_dt.timezone.utc)
    )

    def to_markdown(self) -> str:
        ts = self.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
        fm = [
            "---",
            f"runtime: {self.runtime}",
            f"scope: {self.scope}",
            f"project: {self.project}",
            f"cwd: {self.cwd}",
            f"session_id: {self.session_id}",
            f"timestamp: {ts}",
            f"summary: {self._safe_oneline(self.summary)}",
            "---",
            "",
            self.body.strip(),
            "",
        ]
        return "\n".join(fm)

    @staticmethod
    def from_markdown(text: str) -> "Entry":
        m = re.match(r"---\n(.*?)\n---\n(.*)", text, re.DOTALL)
        if not m:
            raise ValueError("entry missing frontmatter")
        fm_lines = m.group(1).splitlines()
        body = m.group(2).strip()
        fm = {}
        for line in fm_lines:
            if ": " in line:
                k, v = line.split(": ", 1)
                fm[k.strip()] = v.strip()
        ts_str = fm.get("timestamp", "")
        try:
            ts = _dt.datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=_dt.timezone.utc
            )
        except ValueError:
            ts = _dt.datetime.now(_dt.timezone.utc)
        return Entry(
            runtime=fm.get("runtime", ""),
            cwd=fm.get("cwd", ""),
            summary=fm.get("summary", ""),
            body=body,
            session_id=fm.get("session_id", ""),
            scope=fm.get("scope", ""),
            project=fm.get("project", ""),
            timestamp=ts,
        )

    @staticmethod
    def _safe_oneline(s: str) -> str:
        return s.replace("\n", " ").replace("\r", " ").strip()
