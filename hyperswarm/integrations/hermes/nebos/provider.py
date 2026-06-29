"""NebosContextProvider — Hermes session-start Team Nebula company context.

Duck-types Hermes' MemoryProvider ABC. Its only job is to inject a compact
NEBOS company snapshot at the first prefetch of a session (company state changes
slowly, so once/session). No personal recall, no write-back — that distinguishes
it from the personal-instance Jarvis provider. The model still has the live
nebos_* MCP tools for on-demand detail.
"""
from __future__ import annotations

import os

from .nebos_context import company_snapshot, _nebos_creds


class NebosContextProvider:  # duck-types Hermes MemoryProvider ABC
    def __init__(self) -> None:
        self._session_id = ""
        self._hermes_home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
        self._done = False

    @property
    def name(self) -> str:
        return "nebos"

    def is_available(self) -> bool:
        # No network — just confirm NEBOS creds are configured.
        url, token = _nebos_creds(self._hermes_home)
        return bool(url and token)

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._hermes_home = (
            kwargs.get("hermes_home")
            or os.environ.get("HERMES_HOME")
            or os.path.expanduser("~/.hermes")
        )
        self._done = False

    def on_session_switch(self, new_session_id: str, **kwargs) -> None:
        self._session_id = new_session_id
        self._done = False

    def system_prompt_block(self) -> str:
        return (
            "Team Nebula's live company state (pipeline, clients, meetings, alerts) is "
            "auto-provided from NEBOS at the start of each session. Use the nebos_* tools "
            "for deeper or up-to-the-minute company detail."
        )

    def prefetch(self, query: str, *, session_id: str = "", **_) -> str:
        if self._done:
            return ""
        self._done = True
        try:
            return company_snapshot(self._hermes_home)
        except Exception:
            return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        return None

    def sync_turn(self, user_content, assistant_content, *, session_id: str = "", messages=None) -> None:
        return None  # company instance: no write-back

    def on_session_end(self, messages) -> None:
        return None

    def get_tool_schemas(self):
        return []  # live detail comes from the wired nebos_* MCP tools

    def shutdown(self) -> None:
        return None
