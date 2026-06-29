"""Hermes memory-provider plugin entrypoint — Jarvis brain.

Hermes' ``plugins/memory/__init__.py::discover_memory_providers()`` treats a
directory as a provider if its source references ``register_memory_provider``
or ``MemoryProvider``. At load time Hermes calls ``register(ctx)`` and the
provider self-registers via ``ctx.register_memory_provider`` (mirrors the
hyperswarm + retaindb plugins). Only ONE external provider is allowed; this
one is selected by ``memory.provider: jarvis`` in config.yaml.
"""
from __future__ import annotations

import os

from .provider import JarvisMemoryProvider

__all__ = ["JarvisMemoryProvider", "register"]

# Headless one-shot callers (e.g. `hermes -z` synthesis) set this so their
# sessions neither inject memory context nor write observations.
_DISABLE_ENV = "JARVIS_MEMORY_DISABLE"
_TRUTHY = {"1", "true", "yes", "on"}


def register(ctx, provider: "JarvisMemoryProvider | None" = None, **kwargs) -> None:
    if os.environ.get(_DISABLE_ENV, "").strip().lower() in _TRUTHY:
        return
    if provider is None:
        provider = JarvisMemoryProvider()
    ctx.register_memory_provider(provider)
