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


def register(ctx, provider: "HyperSwarmMemoryProvider | None" = None, **kwargs) -> None:
    """Hermes plugin contract: register the HyperSwarm memory provider.

    Hermes loads this plugin and calls ``register(ctx)``; the provider then
    registers itself via ``ctx.register_memory_provider(...)`` — mirrors
    ``plugins/memory/retaindb/__init__.py``. Only ONE external provider is
    allowed.

    ``provider`` is an optional already-built instance (some Hermes builds
    construct the provider and pass it in); when omitted we instantiate the
    default. Extra ``**kwargs`` are accepted/ignored so the entrypoint tolerates
    Hermes passing context kwargs without breaking. The exact signature is
    re-verified against Hermes' ``plugins/memory/__init__.py`` at deploy time
    (Task 5 Step 2 / Task 7).
    """
    if provider is None:
        provider = HyperSwarmMemoryProvider()
    ctx.register_memory_provider(provider)
