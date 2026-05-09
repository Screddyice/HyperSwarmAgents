"""Watchers — long-running daemons that trigger Reflectors / Tuners on session activity.

Watchers replace the cron model: instead of "run every 6h regardless,"
they fire ONLY when an agent actually finishes a session worth processing.
"""
from hyperswarm.watchers.openclaw_sessions import OpenClawSessionWatcher, run_watcher

__all__ = ["OpenClawSessionWatcher", "run_watcher"]
