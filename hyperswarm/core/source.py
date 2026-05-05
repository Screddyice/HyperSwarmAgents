"""Source — captures session state from a specific agent runtime.

A Source plugin is responsible for two things:
  1. install():  set up whatever hook / wrapper / watcher this runtime needs
                 so that capture() gets called automatically. Idempotent.
  2. capture(raw): convert a raw runtime payload into a populated Entry.

Sources never talk to the Store or Sync directly — the orchestrator does that.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from hyperswarm.core.entry import Entry


class Source(ABC):
    name: str = ""  # short identifier used in config.toml (e.g. "claude_code")

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    def install(self) -> None:
        """Set up the auto-capture mechanism for this runtime.

        Implementations should be idempotent — running install() twice should
        not produce duplicate hooks or wrappers.
        """

    @abstractmethod
    def capture(self, raw: dict) -> Entry:
        """Convert a raw runtime payload into an Entry.

        The shape of `raw` depends on the runtime — for example, Claude Code
        Stop hooks pass session info via stdin JSON; Codex wrappers might pass
        a path to a session log. Plugins document their expected shape.
        """
