"""Watchers — long-running daemons that trigger Reflectors / Tuners on session activity.

Watchers replace the cron model: instead of "run every 6h regardless,"
they fire ONLY when an agent actually finishes a session worth processing.
"""
# retired 2026-06-08: openclaw decommissioned, replaced by Hermes memory provider

__all__: list[str] = []
