"""hyperswarm CLI entry point.

Phase 1 ships only the read path (`recent`) and config validation. The write
path (`capture`, `install`, `pull`, `push`) lands in Phase 2 alongside the
Source / Sync reference implementations.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
import sys
from pathlib import Path

try:
    import tomllib  # 3.11+
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from hyperswarm.scopes.path_prefix import PathPrefixScope
from hyperswarm.stores.markdown import MarkdownStore

DEFAULT_CONFIG = "~/.config/hyperswarm/config.toml"


def _load_config(path: str | None) -> dict:
    p = Path(os.path.expanduser(path or DEFAULT_CONFIG))
    if not p.exists():
        return {}
    with p.open("rb") as f:
        return tomllib.load(f)


def _parse_since(s: str) -> _dt.datetime:
    """Parse durations like 24h, 7d, 30m, or an ISO-8601 timestamp."""
    m = re.match(r"^(\d+)([smhd])$", s)
    now = _dt.datetime.now(_dt.timezone.utc)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        seconds = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
        return now - _dt.timedelta(seconds=n * seconds)
    try:
        return _dt.datetime.fromisoformat(s).astimezone(_dt.timezone.utc)
    except ValueError:
        raise SystemExit(f"could not parse --since {s!r}; use 24h, 7d, or ISO-8601")


def cmd_recent(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    store_cfg = cfg.get("store", {}) or {}
    store = MarkdownStore(store_cfg)
    since = _parse_since(args.since)

    entries = list(store.list_since(since))
    entries.sort(key=lambda e: e.timestamp, reverse=True)

    if args.scope:
        entries = [e for e in entries if e.scope == args.scope]
    if args.runtime:
        entries = [e for e in entries if e.runtime == args.runtime]

    if not entries:
        print(f"no entries since {since.isoformat()}")
        return 0

    for e in entries:
        ts = e.timestamp.strftime("%Y-%m-%d %H:%M")
        scope = e.scope or "-"
        print(f"{ts}  [{e.runtime:<14}]  {scope:<10}  {e.summary}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    if not cfg:
        path = args.config or DEFAULT_CONFIG
        print(f"no config at {path} — using built-in defaults")
        return 0
    print(f"loaded config from {args.config or DEFAULT_CONFIG}")

    store_type = (cfg.get("store") or {}).get("type", "markdown")
    if store_type != "markdown":
        print(f"  warning: store.type={store_type!r} not yet implemented in Phase 1")

    scope_cfg = cfg.get("scope") or {}
    if scope_cfg.get("type") in (None, "path_prefix"):
        scope = PathPrefixScope(scope_cfg)
        print(
            f"  scope: path_prefix "
            f"({len(scope_cfg.get('path_prefix', []))} path rules, "
            f"{len(scope_cfg.get('hostname', []))} hostname rules)"
        )
    else:
        print(f"  warning: scope.type={scope_cfg.get('type')!r} not yet implemented in Phase 1")

    sources = cfg.get("source") or []
    print(f"  sources configured: {[s.get('type') for s in sources]} (Phase 2 — not yet wired)")

    syncs = cfg.get("sync") or []
    print(f"  syncs configured: {[s.get('type') for s in syncs]} (Phase 2 — not yet wired)")
    return 0


def _add_config_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--config", default=None,
        help=f"path to config.toml (default {DEFAULT_CONFIG})",
    )


def main() -> int:
    parser = argparse.ArgumentParser(prog="hyperswarm")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_recent = sub.add_parser("recent", help="list recent entries from the local Store")
    _add_config_arg(p_recent)
    p_recent.add_argument("--since", default="7d", help="duration (24h, 7d) or ISO-8601 (default 7d)")
    p_recent.add_argument("--scope", default=None, help="filter by scope tag")
    p_recent.add_argument("--runtime", default=None, help="filter by runtime")
    p_recent.set_defaults(func=cmd_recent)

    p_check = sub.add_parser("check", help="validate config and report what plugins are wired")
    _add_config_arg(p_check)
    p_check.set_defaults(func=cmd_check)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
