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
        return []

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

    def prefetch(self, query: str, *, session_id: str = "", org: str | None = None) -> str:
        active = self._active_org(org)
        since = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=_RECALL_WINDOW_DAYS)
        terms = [w for w in query.lower().split() if w]

        hits: list[tuple[int, Entry]] = []
        for entry in self._store.list_since(since):
            # ORG ISOLATION: skip any entry not visible to the active org.
            if not self._visible(entry, active):
                continue
            body_lc = entry.body.lower()
            score = sum(1 for w in terms if w in body_lc)
            if score:
                hits.append((score, entry))

        if not hits:
            return ""

        hits.sort(key=lambda t: (t[0], t[1].timestamp), reverse=True)
        block = "\n\n".join(
            f"- {entry.body.strip()[:_MAX_ENTRY_CHARS]}" for _, entry in hits[:_TOP_K]
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
        raise NotImplementedError(tool_name)
