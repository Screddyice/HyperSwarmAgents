"""HyperSwarmMemoryProvider — Hermes working memory backed by HyperSwarm.

Duck-types Hermes' ``agent/memory_provider.py::MemoryProvider`` ABC. We do not
import the Hermes ABC here so the provider remains testable in the
HyperSwarmAgents venv; at runtime inside Hermes' venv it is registered via
``ctx.register_memory_provider`` (see ``__init__.py``) and satisfies the ABC
structurally.

Memory model
------------
- Recall (``prefetch``): list recent HyperSwarm entries, filter by ORG SCOPE
  (active org + shared/unscoped only — the security boundary), rank by keyword
  overlap, inject the top-K as a memory-context block.
- Write (``on_session_end`` / selective ``sync_turn``): ``sync_turn`` buffers
  only and writes NOTHING per-turn (honors the HyperSwarm capture-scope rule —
  the store holds session-left-off + learnings, never every turn). The real
  write is a single scope-tagged session-left-off entry at ``on_session_end``.

Org tag carrier
---------------
The org/company tag is carried by ``Entry.scope`` (a plain string, e.g.
"TMN"/"Cliqk"/"TRC"), NOT a ``frontmatter`` dict and NOT an ``Entry.company``
field — see ``hyperswarm/core/{entry,scope}.py``. An empty ``scope`` means the
entry is shared and visible to every org.
"""
from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path

from hyperswarm.core.entry import Entry
from hyperswarm.stores.markdown import MarkdownStore

# Recall window: how far back ``prefetch`` scans for keyword matches.
_RECALL_WINDOW_DAYS = 365
_TOP_K = 5
_MAX_ENTRY_CHARS = 500

# RECENCY weighting. Ranking blends keyword overlap with a recency component so
# casual/vague queries ("what was I working on?") surface the NEWEST left-off
# instead of an older keyword-rich entry. The recency component is bounded to
# (0, 1] (newest -> ~1, decaying with age) and weighted by ``_RECENCY_WEIGHT``.
#
# Calibration: with _RECENCY_WEIGHT just above 1, a single extra keyword match
# (the typical vague-phrasing case, e.g. "working on" matching one more low-
# signal word) does NOT outrank a much newer entry — recency is the dominant
# tiebreak when keyword scores are close. But a SPECIFIC query whose unique term
# matches ONLY the old entry still returns it: the other entries score 0 and are
# filtered out entirely, so recency never has the chance to bury a unique hit.
# A genuinely keyword-richer match (2+ more terms) still beats mere recency.
# Deterministic, no external deps.
_RECENCY_WEIGHT = 1.1
_RECENCY_HALFLIFE_DAYS = 30.0


class HyperSwarmMemoryProvider:  # duck-types Hermes MemoryProvider ABC
    def __init__(self, root: str | None = None):
        self._store = MarkdownStore({"path": root} if root else None)
        self._root = Path(os.path.expanduser(root)) if root else self._store.root
        self._session_id = ""
        self._org: str | None = None
        self._buffer: list[tuple[str, str]] = []

    # --- identity / lifecycle ---------------------------------------------

    @property
    def name(self) -> str:
        return "hyperswarm"

    def is_available(self) -> bool:
        return (self._root / "entries").exists()

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        # Active org may arrive as ``org`` or under Hermes' agent_context.
        self._org = kwargs.get("org")
        if self._org is None:
            ctx = kwargs.get("agent_context") or {}
            if isinstance(ctx, dict):
                self._org = ctx.get("org") or ctx.get("company")
        self._buffer = []

    def get_tool_schemas(self):
        # One explicit recall tool. The description deliberately routes
        # deep/personal/health/broad queries to the corpus-mcp brain so the
        # agent keeps HyperSwarm for working memory only.
        return [
            {
                "name": "hyperswarm_search",
                "description": (
                    "Search Shawn's HyperSwarm working memory (org-scoped, "
                    "current-context). Use this for recent working memory: what "
                    "was last left off on, recent decisions, session learnings, "
                    "and short-term project state. Do NOT use this for deep, "
                    "personal, health, or broad knowledge queries — for those, "
                    "use the corpus-mcp brain tool instead (it holds the deep, "
                    "org-isolated long-term knowledge). Keep HyperSwarm for "
                    "working memory; route depth to corpus."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Keywords to search working memory for.",
                        }
                    },
                    "required": ["query"],
                },
            }
        ]

    def shutdown(self) -> None:
        self._buffer = []

    # --- org scope helpers (the security boundary) ------------------------

    @staticmethod
    def _entry_company(entry: Entry) -> str | None:
        """Return the org tag for an entry, or None if shared/unscoped."""
        scope = (getattr(entry, "scope", "") or "").strip()
        return scope or None

    def _active_org(self, override: str | None = None) -> str | None:
        return override if override is not None else self._org

    def _visible(self, entry: Entry, active: str | None) -> bool:
        """An entry is visible iff it is shared (no org) OR matches active org.

        If no active org is set, everything is visible (transitional state —
        org-sensitive depth lives in the already-isolated corpus, not here).
        """
        company = self._entry_company(entry)
        if company is None:
            return True  # shared entries are visible to every org
        if active is None:
            return True
        return company == active

    # --- recall -----------------------------------------------------------

    @staticmethod
    def _recency_score(entry: Entry, now: _dt.datetime) -> float:
        """Bounded recency component in (0, 1]: newest ~1, decaying with age.

        Exponential half-life decay keyed on ``entry.timestamp``. Deterministic
        and dependency-free. Future-dated or now -> ~1.0; ages monotonically.
        """
        ts = getattr(entry, "timestamp", None)
        if ts is None:
            return 0.0
        age_days = max(0.0, (now - ts).total_seconds() / 86400.0)
        return 0.5 ** (age_days / _RECENCY_HALFLIFE_DAYS)

    def prefetch(self, query: str, *, session_id: str = "", org: str | None = None) -> str:
        active = self._active_org(org)
        now = _dt.datetime.now(_dt.timezone.utc)
        since = now - _dt.timedelta(days=_RECALL_WINDOW_DAYS)
        terms = [w for w in query.lower().split() if w]

        # Blended ranking: keyword_overlap + _RECENCY_WEIGHT * recency. With
        # _RECENCY_WEIGHT < 1, one extra UNIQUE keyword match (a specific query)
        # outranks a merely-newer entry; when keyword scores tie or are near
        # zero (vague query) recency dominates. Carry timestamp as a final
        # deterministic tiebreak.
        hits: list[tuple[float, _dt.datetime, Entry]] = []
        for entry in self._store.list_since(since):
            # ORG ISOLATION: skip any entry not visible to the active org.
            if not self._visible(entry, active):
                continue
            body_lc = entry.body.lower()
            keyword_score = sum(1 for w in terms if w in body_lc)
            if not keyword_score:
                continue
            blended = keyword_score + _RECENCY_WEIGHT * self._recency_score(entry, now)
            hits.append((blended, entry.timestamp, entry))

        if not hits:
            return ""

        hits.sort(key=lambda t: (t[0], t[1]), reverse=True)
        block = "\n\n".join(
            f"- {entry.body.strip()[:_MAX_ENTRY_CHARS]}" for _, _, entry in hits[:_TOP_K]
        )
        return (
            "<memory-context>\n"
            "Relevant HyperSwarm memories:\n"
            f"{block}\n"
            "</memory-context>"
        )

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        # No background worker yet; recall is synchronous in ``prefetch``.
        return None

    def system_prompt_block(self) -> str:
        return ""

    # --- write path (capture-scope safe) ----------------------------------

    def sync_turn(self, user_content, assistant_content, *, session_id: str = "", messages=None) -> None:
        # Buffer only. NEVER write a per-turn entry — the store holds
        # session-left-off + learnings, not every turn.
        self._buffer.append((str(user_content), str(assistant_content)))

    def on_session_end(self, messages) -> None:
        if not self._buffer:
            return
        last = self._buffer[-3:]
        body = "## Session left-off\n" + "\n".join(
            f"- {u} -> {a}" for u, a in last
        )
        entry = Entry(
            runtime="hermes",
            cwd=os.getcwd(),
            summary="Hermes session left-off",
            body=body,
            session_id=self._session_id,
            scope=self._org or "",  # org tag carried by Entry.scope
        )
        self._store.write(entry)
        self._buffer = []

    def handle_tool_call(self, tool_name, args, **kwargs):
        import json

        if tool_name != "hyperswarm_search":
            raise NotImplementedError(tool_name)
        args = args or {}
        query = args.get("query", "")
        # Reuse the org-isolated recall path so the tool inherits the same
        # security boundary as prefetch(). prefetch() returns a memory-context
        # block whose hit lines are prefixed with "- "; parse those back out.
        block = self.prefetch(query, org=self._org)
        results = [
            line[2:] for line in block.splitlines() if line.startswith("- ")
        ]
        return json.dumps({"results": results})
