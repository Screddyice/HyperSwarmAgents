"""Sync — moves entries between nodes.

When capture happens off the canonical host (e.g. on a remote OpenClaw box),
a Sync plugin is responsible for moving those entries to wherever the
canonical Store lives. Sync is a one-way push (or pull); bi-directional
syncing is composed from two Sync plugins.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class Sync(ABC):
    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    def push(self) -> int:
        """Push local entries to the configured remote. Return count moved."""

    @abstractmethod
    def pull(self) -> int:
        """Pull remote entries to the local Store. Return count moved."""
