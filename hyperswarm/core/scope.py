"""Scope — tags each Entry so reads can filter.

Scope sits between Source.capture() (which produces an Entry without a scope)
and Store.write() (which persists it with a scope tag). The orchestrator calls
Scope.tag(entry) before writing.

Common scope axes: project, team, company, environment. The Entry doesn't
care what the axis is — it just stores the resolved tag string.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from hyperswarm.core.entry import Entry


class Scope(ABC):
    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    def tag(self, entry: Entry) -> str:
        """Return the scope tag for this entry. May inspect entry.cwd, env vars,
        the host's identity — whatever makes sense for the deployment.

        If no rule matches, return a sentinel like "" or "unscoped". Reads can
        choose to filter those out or include them in the unfiltered view.
        """
