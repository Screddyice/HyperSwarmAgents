"""Reflectors — distill raw session activity into curated memories.

A reflector reads a runtime's session logs, calls an LLM to extract
high-signal learnings, and writes them as markdown files into a
designated memory dir. Unlike Sources (which capture-and-store), Reflectors
synthesize across sessions to make the brain smarter over time.

Pattern reference: Park et al, "Generative Agents: Interactive Simulacra
of Human Behavior" (2023) — Memory Stream → Reflection → Retrieval → Planning.
This module implements the Reflection layer.
"""
from hyperswarm.reflectors.openclaw_session import (
    OpenClawSessionReflector,
    reflect_agent,
)

__all__ = ["OpenClawSessionReflector", "reflect_agent"]
