"""Hermes memory-provider plugin entrypoint — NEBOS company context.

Injects a session-start Team Nebula company snapshot (pipeline, clients,
meetings, alerts) from NEBOS as the context layer. Read-only context; no
personal recall, no write-back. Selected via ``memory.provider: nebos``.
"""
from __future__ import annotations

import os

from .provider import NebosContextProvider

__all__ = ["NebosContextProvider", "register"]

_DISABLE_ENV = "NEBOS_MEMORY_DISABLE"
_TRUTHY = {"1", "true", "yes", "on"}


def register(ctx, provider=None, **kwargs) -> None:
    if os.environ.get(_DISABLE_ENV, "").strip().lower() in _TRUTHY:
        return
    if provider is None:
        provider = NebosContextProvider()
    ctx.register_memory_provider(provider)
