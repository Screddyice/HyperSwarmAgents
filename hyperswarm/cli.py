"""hyperswarm CLI entry point."""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import sys
from pathlib import Path

try:
    import tomllib  # 3.11+
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from hyperswarm.scopes.git_remote import GitRemoteScope
from hyperswarm.scopes.path_prefix import PathPrefixScope

_SCOPE_REGISTRY = {
    "path_prefix": PathPrefixScope,
    "git_remote": GitRemoteScope,
}
from hyperswarm.sources import SOURCE_REGISTRY
from hyperswarm.stores.markdown import MarkdownStore
from hyperswarm.syncs import SYNC_REGISTRY

DEFAULT_CONFIG = "~/.config/hyperswarm/config.toml"


def _load_config(path: str | None) -> dict:
    p = Path(os.path.expanduser(path or DEFAULT_CONFIG))
    if not p.exists():
        return {}
    with p.open("rb") as f:
        return tomllib.load(f)


def _build_store(cfg: dict) -> MarkdownStore:
    store_cfg = cfg.get("store", {}) or {}
    store_type = store_cfg.get("type", "markdown")
    if store_type != "markdown":
        raise SystemExit(f"store.type={store_type!r} is not yet implemented; use 'markdown'")
    return MarkdownStore(store_cfg)


def _build_scope(cfg: dict):
    scope_cfg = cfg.get("scope", {}) or {}
    scope_type = scope_cfg.get("type", "path_prefix")
    if scope_type not in _SCOPE_REGISTRY:
        known = sorted(_SCOPE_REGISTRY)
        raise SystemExit(f"scope.type={scope_type!r} is not implemented; known: {known}")
    return _SCOPE_REGISTRY[scope_type](scope_cfg)


def _find_source_config(cfg: dict, runtime: str) -> dict:
    """Return the [[source]] block matching the given --runtime, or {}.

    Sources keyed by either their canonical name (`claude_code`) or its
    runtime-friendly hyphenated form (`claude-code`) are equivalent — the
    user can write either in config.toml.
    """
    for s in cfg.get("source") or []:
        if s.get("type") in (runtime, runtime.replace("-", "_"), runtime.replace("_", "-")):
            return s
    return {}


def _parse_since(s: str) -> _dt.datetime:
    m = re.match(r"^(\d+)([smhd])$", s)
    now = _dt.datetime.now(_dt.timezone.utc)
    if m:
        n = int(m.group(1))
        seconds = {"s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]
        return now - _dt.timedelta(seconds=n * seconds)
    try:
        return _dt.datetime.fromisoformat(s).astimezone(_dt.timezone.utc)
    except ValueError:
        raise SystemExit(f"could not parse --since {s!r}; use 24h, 7d, or ISO-8601")


# ----------------------------------------------------------------- recent
def cmd_recent(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    store = _build_store(cfg)
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
        project = e.project or "-"
        print(f"{ts}  [{e.runtime:<14}]  {scope:<10}  {project:<22}  {e.summary}")
    return 0


# ----------------------------------------------------------------- capture
def cmd_capture(args: argparse.Namespace) -> int:
    """Read raw runtime payload from stdin (JSON or empty), dispatch to the
    Source, tag with Scope, write to Store. Failures are logged but never
    raise — a hook firing this command must not crash the user's session.
    """
    cfg = _load_config(args.config)
    runtime = args.runtime
    if runtime not in SOURCE_REGISTRY:
        print(f"unknown runtime {runtime!r}; known: {sorted(SOURCE_REGISTRY)}", file=sys.stderr)
        return 1

    source_cfg = _find_source_config(cfg, runtime)
    Source = SOURCE_REGISTRY[runtime]
    source = Source(source_cfg)

    raw_text = ""
    if not sys.stdin.isatty():
        try:
            raw_text = sys.stdin.read()
        except Exception:
            raw_text = ""
    raw: dict = {}
    if raw_text.strip():
        try:
            raw = json.loads(raw_text)
        except json.JSONDecodeError:
            # Don't fail on malformed input — capture from {} and let the
            # source produce whatever sensible Entry it can.
            raw = {}

    try:
        entry = source.capture(raw)
    except Exception as e:
        print(f"capture failed for runtime={runtime}: {e}", file=sys.stderr)
        return 1
    if entry is None:
        # OpenClawSource returns None when the queue is empty — that's
        # success, not failure.
        return 0

    try:
        scope = _build_scope(cfg)
        entry.scope = scope.tag(entry)
        store = _build_store(cfg)
        sid = store.write(entry)
        if args.verbose:
            print(f"wrote {sid}")
    except Exception as e:
        print(f"persist failed for runtime={runtime}: {e}", file=sys.stderr)
        return 1
    return 0


# ----------------------------------------------------------------- install
def cmd_install(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)

    # If --runtime is given, install just that one. Otherwise install every
    # runtime in config.
    targets = []
    if args.runtime:
        if args.runtime not in SOURCE_REGISTRY:
            print(f"unknown runtime {args.runtime!r}; known: {sorted(SOURCE_REGISTRY)}", file=sys.stderr)
            return 1
        source_cfg = _find_source_config(cfg, args.runtime)
        targets.append((args.runtime, source_cfg))
    else:
        seen = set()
        for s in cfg.get("source") or []:
            t = s.get("type")
            if t in SOURCE_REGISTRY and t not in seen:
                targets.append((t, s))
                seen.add(t)

    if not targets:
        print("no sources to install — pass --runtime X or add [[source]] blocks to config.toml")
        return 1

    rc = 0
    for name, source_cfg in targets:
        Source = SOURCE_REGISTRY[name]
        source = Source(source_cfg)
        try:
            source.install()
            print(f"installed {name}")
        except Exception as e:
            print(f"install failed for {name}: {e}", file=sys.stderr)
            rc = 1
    return rc


# ----------------------------------------------------------------- push/pull
def _build_syncs(cfg: dict) -> list:
    """Instantiate every [[sync]] block whose type is in SYNC_REGISTRY."""
    out = []
    for s in cfg.get("sync") or []:
        t = s.get("type")
        if t in SYNC_REGISTRY:
            out.append((t, SYNC_REGISTRY[t](s)))
    return out


def cmd_push(args: argparse.Namespace) -> int:
    """Run push() on every configured Sync. Each Sync's own only_on_host gate
    decides whether it actually fires on this host — push() returns 0 if the
    gate excludes us, so cron can run this everywhere safely.
    """
    cfg = _load_config(args.config)
    syncs = _build_syncs(cfg)
    if not syncs:
        if args.verbose:
            print("no syncs configured")
        return 0
    rc = 0
    for name, sync in syncs:
        try:
            n = sync.push()
            if args.verbose:
                print(f"{name}: pushed {n} item(s)")
        except Exception as e:
            print(f"{name}: push failed: {e}", file=sys.stderr)
            rc = 1
    return rc


def cmd_pull(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    syncs = _build_syncs(cfg)
    if not syncs:
        if args.verbose:
            print("no syncs configured")
        return 0
    rc = 0
    for name, sync in syncs:
        try:
            n = sync.pull()
            if args.verbose:
                print(f"{name}: pulled {n} item(s)")
        except Exception as e:
            print(f"{name}: pull failed: {e}", file=sys.stderr)
            rc = 1
    return rc


# ----------------------------------------------------------------- check
def cmd_check(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    if not cfg:
        path = args.config or DEFAULT_CONFIG
        print(f"no config at {path} — using built-in defaults")
        return 0
    print(f"loaded config from {args.config or DEFAULT_CONFIG}")

    store_type = (cfg.get("store") or {}).get("type", "markdown")
    if store_type != "markdown":
        print(f"  warning: store.type={store_type!r} not yet implemented")

    scope_cfg = cfg.get("scope") or {}
    scope_type = scope_cfg.get("type") or "path_prefix"
    if scope_type == "path_prefix":
        print(
            f"  scope: path_prefix "
            f"({len(scope_cfg.get('path_prefix', []))} path rules, "
            f"{len(scope_cfg.get('hostname', []))} hostname rules)"
        )
    elif scope_type == "git_remote":
        print(
            f"  scope: git_remote "
            f"({len(scope_cfg.get('git_remote', []))} org rules, "
            f"{len(scope_cfg.get('path_prefix', []))} path-fallback rules, "
            f"{len(scope_cfg.get('hostname', []))} hostname rules)"
        )
    else:
        print(f"  warning: scope.type={scope_type!r} not yet implemented")

    sources = cfg.get("source") or []
    known_sources = [s.get("type") for s in sources if s.get("type") in SOURCE_REGISTRY]
    unknown_sources = [s.get("type") for s in sources if s.get("type") not in SOURCE_REGISTRY]
    print(f"  sources wired: {known_sources}")
    if unknown_sources:
        print(f"  sources unknown to this version: {unknown_sources}")

    syncs = cfg.get("sync") or []
    known_syncs = [s.get("type") for s in syncs if s.get("type") in SYNC_REGISTRY]
    unknown_syncs = [s.get("type") for s in syncs if s.get("type") not in SYNC_REGISTRY]
    print(f"  syncs wired: {known_syncs}")
    if unknown_syncs:
        print(f"  syncs unknown to this version: {unknown_syncs}")
    return 0


# ----------------------------------------------------------------- main
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

    p_capture = sub.add_parser(
        "capture",
        help="capture a session from a runtime — reads raw payload from stdin",
    )
    _add_config_arg(p_capture)
    p_capture.add_argument("--runtime", required=True, help="source plugin name (claude_code | codex | openclaw)")
    p_capture.add_argument("--verbose", "-v", action="store_true")
    p_capture.set_defaults(func=cmd_capture)

    p_install = sub.add_parser(
        "install",
        help="install hooks/wrappers/watchers for one or all configured sources",
    )
    _add_config_arg(p_install)
    p_install.add_argument("--runtime", default=None, help="install just this source (defaults to all in config)")
    p_install.set_defaults(func=cmd_install)

    p_push = sub.add_parser("push", help="run push() on every configured Sync")
    _add_config_arg(p_push)
    p_push.add_argument("--verbose", "-v", action="store_true")
    p_push.set_defaults(func=cmd_push)

    p_pull = sub.add_parser("pull", help="run pull() on every configured Sync")
    _add_config_arg(p_pull)
    p_pull.add_argument("--verbose", "-v", action="store_true")
    p_pull.set_defaults(func=cmd_pull)

    p_check = sub.add_parser("check", help="validate config and report what plugins are wired")
    _add_config_arg(p_check)
    p_check.set_defaults(func=cmd_check)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
