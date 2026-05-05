"""Store — persists Entry records.

The Store is intentionally narrow — write/read/list. Filtering / ranking is
done in the orchestrator so multiple Store implementations don't have to
re-implement the same query logic.
"""
from __future__ import annotations

import datetime as _dt
from abc import ABC, abstractmethod
from typing import Iterable

from hyperswarm.core.entry import Entry


class Store(ABC):
    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    def write(self, entry: Entry) -> str:
        """Persist an entry. Return an opaque storage id (path, uuid, etc.)."""

    @abstractmethod
    def read(self, storage_id: str) -> Entry:
        """Load a single entry by id."""

    @abstractmethod
    def list_since(self, since: _dt.datetime) -> Iterable[Entry]:
        """Yield entries with timestamp >= since, in any order.

        Most callers will sort the result by timestamp — Store impls are not
        required to return sorted results.
        """
