"""Hermes memory-provider plugin entrypoint.

Hermes' ``plugins/memory/__init__.py::discover_memory_providers()`` treats a
directory as a provider if its source references ``register_memory_provider``
or ``MemoryProvider``. At load time Hermes calls ``register(ctx)`` and the
provider registers itself via ``ctx.register_memory_provider`` (mirrors
``plugins/memory/retaindb/__init__.py``). Only ONE external provider is allowed.
"""
from __future__ import annotations

from .provider import HyperSwarmMemoryProvider

__all__ = ["HyperSwarmMemoryProvider", "register"]


def register(ctx) -> None:
    """Hermes plugin contract: register the HyperSwarm memory provider.

    The exact entrypoint signature is verified against Hermes'
    ``plugins/memory/__init__.py`` at deploy time (Task 5/7).
    """
    ctx.register_memory_provider(HyperSwarmMemoryProvider())
