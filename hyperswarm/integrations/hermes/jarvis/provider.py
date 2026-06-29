"""JarvisMemoryProvider — Hermes memory backed by the Jarvis brain.

Duck-types Hermes' ``agent/memory_provider.py::MemoryProvider`` ABC (not
imported, so the module stays importable outside Hermes' venv; at runtime it
is registered via ``ctx.register_memory_provider`` — see ``__init__.py`` — and
satisfies the ABC structurally). Replaces the HyperSwarm working-memory
provider on Shawn's personal Hermes.

Memory model
------------
- Recall (``prefetch``): on each turn, semantic search the Jarvis brain
  (``POST /search`` against the shawn-corpus REST API, pgvector) and inject the
  top-K observation texts as RAW context (the MemoryManager wraps it in
  ``<memory-context>`` — we must NOT pre-wrap). At the first prefetch of a
  session, optionally prepend today's digest. Hard timeout → return "" so a
  slow/unreachable brain never stalls a turn.
- Write (``on_session_end``): buffer turns in-session (``sync_turn`` writes
  nothing per-turn — honors the corpus capture-scope rule), then write ONE
  distilled observation. ``write_path``:
    * ``dual``  (default) — write a HyperSwarm markdown entry (keeps the
      learnings-classifier feed) AND a SYNCHRONOUS ``POST /observations`` with
      the SAME ``source_entry_path`` so the later ingestion collapses onto the
      same PG row (read-after-write, no duplicate).
    * ``rest``  — synchronous ``POST /observations`` only (fast; skips the
      learnings classifier).
    * ``hyperswarm`` — markdown entry only (today's behavior; async ~30s).

query_brain (the fine-tuned MLX adapter) is intentionally NOT wrapped here — it
has no REST route and is already exposed by the wired ``corpus-mcp`` MCP. We
only nudge the model toward it in ``system_prompt_block``.
"""
from __future__ import annotations

import json
import os
import uuid
import datetime as _dt
import urllib.request
import urllib.error
from pathlib import Path

# Markdown leg (dual / hyperswarm write paths). Available in the gateway venv.
try:
    from hyperswarm.core.entry import Entry
    from hyperswarm.stores.markdown import MarkdownStore
    _HS_OK = True
except Exception:  # pragma: no cover — only the rest-only path works without it
    _HS_OK = False

# NEBOS company-context layer (session-start snapshot of Team Nebula state).
try:
    from .nebos_context import company_snapshot
    _NEBOS_OK = True
except Exception:
    _NEBOS_OK = False

_DEFAULTS = {
    "base_url": "http://127.0.0.1:8210",
    "include_nebos_context": True,
    "search_k": 6,
    "recall_char_limit": 2200,
    "entry_char_limit": 500,
    "include_today_digest": True,
    "write_path": "dual",          # dual | rest | hyperswarm
    "search_timeout": 2.5,         # hot-path recall: fail fast
    "write_timeout": 6.0,
    "company": "personal",
}


def _load_cfg() -> dict:
    """Read the optional ``jarvis:`` block from $HERMES_HOME/config.yaml; fall
    back to defaults. Best-effort — never raises."""
    cfg = dict(_DEFAULTS)
    home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    path = os.path.join(home, "config.yaml")
    try:
        import yaml  # PyYAML is present in the Hermes venv
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        block = data.get("jarvis") or {}
        if isinstance(block, dict):
            for k in _DEFAULTS:
                if k in block and block[k] is not None:
                    cfg[k] = block[k]
    except Exception:
        pass
    return cfg


class JarvisMemoryProvider:  # duck-types Hermes MemoryProvider ABC
    def __init__(self) -> None:
        self._cfg = _load_cfg()
        self._token = os.environ.get("JARVIS_API_TOKEN", "")
        self._session_id = ""
        self._buffer: list[tuple[str, str]] = []
        self._digest_done = False

    # --- identity / lifecycle --------------------------------------------

    @property
    def name(self) -> str:
        return "jarvis"

    def is_available(self) -> bool:
        # No network here — just config + creds (ABC contract).
        return bool(self._token and self._cfg.get("base_url"))

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._buffer = []
        self._digest_done = False
        self._nebos_done = False
        self._hermes_home = (
            kwargs.get("hermes_home")
            or os.environ.get("HERMES_HOME")
            or os.path.expanduser("~/.hermes")
        )
        # Re-read token in case the env was populated after construction.
        if not self._token:
            self._token = os.environ.get("JARVIS_API_TOKEN", "")

    def on_session_switch(self, new_session_id: str, **kwargs) -> None:
        self._session_id = new_session_id
        self._buffer = []
        self._digest_done = False

    def shutdown(self) -> None:
        self._buffer = []

    # --- HTTP helpers (stdlib only) --------------------------------------

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"}

    def _post(self, path: str, payload: dict, timeout: float):
        url = self._cfg["base_url"].rstrip("/") + path
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers=self._headers(), method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)

    def _get(self, path: str, timeout: float):
        url = self._cfg["base_url"].rstrip("/") + path
        req = urllib.request.Request(url, headers=self._headers(), method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)

    # --- recall ----------------------------------------------------------

    def system_prompt_block(self) -> str:
        return (
            "Your long-term memory is Jarvis (Shawn's brain). Relevant memories are "
            "auto-recalled each turn. Use the `jarvis_recall` tool for deeper or older "
            "lookups, the `coding_leftoff` tool when he asks where he left off / what he was "
            "working on / a repo's state (claude-mem coding handoffs), and the corpus-mcp "
            "`query_brain` tool for judgment calls in Shawn's voice."
        )

    def _hits_to_lines(self, hits: list) -> list[str]:
        lines = []
        cap = int(self._cfg["entry_char_limit"])
        for h in hits or []:
            obs = h.get("obs") or {}
            text = (obs.get("text") if isinstance(obs, dict) else None) or h.get("excerpt") or ""
            text = " ".join(str(text).split())
            if text:
                lines.append(f"- {text[:cap]}")
        return lines

    def prefetch(self, query: str, *, session_id: str = "", **_) -> str:
        if not query or not query.strip() or not self.is_available():
            return ""
        parts: list[str] = []
        # NEBOS company-context layer: inject a Team Nebula snapshot once per
        # session (first prefetch). Self-fail-soft; never stalls the turn.
        if _NEBOS_OK and self._cfg.get("include_nebos_context") and not getattr(self, "_nebos_done", False):
            self._nebos_done = True
            try:
                snap = company_snapshot(getattr(self, "_hermes_home", None) or "")
                if snap:
                    parts.append(snap)
            except Exception:
                pass
        try:
            if self._cfg.get("include_today_digest") and not self._digest_done:
                self._digest_done = True
                try:
                    dig = self._get("/digests/today", float(self._cfg["search_timeout"]))
                    summ = (dig or {}).get("summary") if isinstance(dig, dict) else None
                    if summ:
                        parts.append("Today: " + " ".join(str(summ).split())[:500])
                except Exception:
                    pass
            res = self._post(
                "/search",
                {"query": query, "k": int(self._cfg["search_k"])},
                float(self._cfg["search_timeout"]),
            )
            lines = self._hits_to_lines((res or {}).get("hits", []))
            if lines:
                parts.append("Relevant memories from Jarvis:\n" + "\n".join(lines))
        except Exception:
            # Brain slow/down — never stall the turn.
            return "\n\n".join(parts).strip()
        block = "\n\n".join(parts).strip()
        cap = int(self._cfg["recall_char_limit"])
        return block[:cap]

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        return None

    # --- write path ------------------------------------------------------

    def sync_turn(self, user_content, assistant_content, *, session_id: str = "", messages=None) -> None:
        # Buffer only — never write a per-turn observation.
        self._buffer.append((str(user_content), str(assistant_content)))

    def _distilled_body(self) -> str:
        last = self._buffer[-3:]
        return "## Hermes session\n" + "\n".join(f"- {u} -> {a}" for u, a in last)

    def on_session_end(self, messages) -> None:
        if not self._buffer:
            return
        body = self._distilled_body()
        wp = str(self._cfg.get("write_path", "dual"))
        entry_path = None
        try:
            if wp in ("dual", "hyperswarm") and _HS_OK:
                entry = Entry(
                    runtime="hermes",
                    cwd=os.getcwd(),
                    summary="Hermes session (Jarvis-backed)",
                    body=body,
                    session_id=self._session_id,
                    scope=str(self._cfg.get("company") or ""),
                )
                entry_path = MarkdownStore().write(entry)  # returns str(path); sets entry.storage_id
            if wp in ("dual", "rest"):
                payload = {
                    "text": body,
                    "company": self._cfg.get("company"),
                    "tags": ["hermes", "assistant-session"],
                }
                if entry_path:
                    # Match the path the ingestion daemon records (entry.storage_id)
                    # so record_observation collapses onto the same PG row.
                    payload["source_entry_path"] = entry_path
                    payload["idempotency_key"] = str(uuid.uuid5(uuid.NAMESPACE_URL, entry_path))
                else:
                    payload["idempotency_key"] = str(
                        uuid.uuid5(uuid.NAMESPACE_URL, f"{self._session_id}:{body[:200]}")
                    )
                self._post("/observations", payload, float(self._cfg["write_timeout"]))
        except Exception:
            # Never raise on session end.
            pass
        finally:
            self._buffer = []

    # --- tools -----------------------------------------------------------

    def get_tool_schemas(self):
        return [
            {
                "name": "jarvis_recall",
                "description": (
                    "Semantic search of Shawn's long-term Jarvis brain (observations, "
                    "decisions, facts; pgvector). Use for deeper or older recall than the "
                    "auto-injected memory. For judgment calls in Shawn's voice, use the "
                    "corpus-mcp query_brain tool instead."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to recall."},
                        "k": {"type": "integer", "description": "Max results (default 6)."},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "coding_leftoff",
                "description": (
                    "Find where Shawn last left off on a CODING project, from his claude-mem "
                    "session handoffs in HyperSwarm (project, cwd, what he was doing, and the "
                    "next steps). Use when he asks 'where did I leave off', 'what was I working "
                    "on', or about the state of a specific repo/project. `query` = a project or "
                    "repo name or topic; leave empty for his most recent coding sessions."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Project/repo name or topic; empty = most recent."},
                        "k": {"type": "integer", "description": "Max sessions (default 5)."},
                    },
                },
            },
        ]

    def _coding_leftoff(self, query: str, k: int = 5, window_days: int = 120) -> list[dict]:
        """Recent claude-mem coding sessions from the HyperSwarm store, ranked by
        query-keyword overlap then recency. Returns formatted dicts."""
        if not _HS_OK:
            return []
        try:
            store = MarkdownStore()
            since = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=window_days)
            terms = [t for t in (query or "").lower().split() if t]
            cands = []
            for e in store.list_since(since):
                rt = (getattr(e, "runtime", "") or "")
                body = e.body or ""
                bl = body.lower()
                # claude-mem coding sessions (distilled session / leftoff / lesson entries)
                if "claude-mem" not in rt and "left off" not in bl and "next_steps" not in bl and "request:" not in bl:
                    continue
                hay = (bl + " " + (getattr(e, "project", "") or "").lower()
                       + " " + (getattr(e, "cwd", "") or "").lower())
                score = sum(1 for t in terms if t in hay)
                if terms and score == 0:
                    continue  # a query was given but nothing matched
                cands.append((score, e.timestamp, e))
            # query given -> rank by score then recency; no query -> most recent
            cands.sort(key=lambda x: (x[0], x[1]), reverse=True)
            out = []
            for _s, _ts, e in cands[:k]:
                out.append({
                    "project": getattr(e, "project", "") or "",
                    "cwd": getattr(e, "cwd", "") or "",
                    "when": str(e.timestamp)[:16],
                    "text": " ".join((e.body or "").split())[:600],
                })
            return out
        except Exception:
            return []

    def handle_tool_call(self, tool_name, args, **kwargs):
        args = args or {}
        if tool_name == "coding_leftoff":
            results = self._coding_leftoff(args.get("query", ""), int(args.get("k") or 5))
            return json.dumps({"results": results})
        if tool_name != "jarvis_recall":
            raise NotImplementedError(tool_name)
        query = args.get("query", "")
        k = int(args.get("k") or self._cfg["search_k"])
        if not query.strip() or not self.is_available():
            return json.dumps({"results": []})
        try:
            res = self._post("/search", {"query": query, "k": k}, float(self._cfg["write_timeout"]))
        except Exception as e:
            return json.dumps({"results": [], "error": f"{type(e).__name__}: {e}"})
        results = []
        for h in (res or {}).get("hits", []):
            obs = h.get("obs") or {}
            text = (obs.get("text") if isinstance(obs, dict) else None) or h.get("excerpt") or ""
            results.append({"text": " ".join(str(text).split()), "score": h.get("score"), "id": str(h.get("id"))})
        return json.dumps({"results": results})
