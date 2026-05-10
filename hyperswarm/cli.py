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

    p_watch = sub.add_parser(
        "watch",
        help="long-running daemon that fires reflect+tune on session-end (replaces per-agent cron)",
    )
    p_watch.add_argument(
        "--agent",
        action="append",
        required=True,
        help="agent id to watch (repeat for multiple: --agent jarvis --agent clawdbot)",
    )
    p_watch.add_argument("--poll-interval", type=int, default=None, help="seconds between scans (default 30)")
    p_watch.add_argument(
        "--debounce",
        type=int,
        default=None,
        help="seconds of session idle before triggering (default 300 = 5 min)",
    )
    p_watch.add_argument("--no-tune", action="store_true", help="run reflect only, skip tune-collect/trigger")
    p_watch.set_defaults(func=cmd_watch)

    p_tune_collect = sub.add_parser(
        "tune-collect",
        help="append new user/assistant pairs from agent sessions to the fine-tune corpus",
    )
    p_tune_collect.add_argument("--agent", required=True)
    p_tune_collect.add_argument("--host", default=None)
    p_tune_collect.add_argument("--verbose", "-v", action="store_true")
    p_tune_collect.set_defaults(func=cmd_tune_collect)

    p_tune_ptp = sub.add_parser(
        "tune-pull-train-push",
        help="Mac-side helper: pull a server's corpus, train locally with MLX, push the adapter back",
    )
    p_tune_ptp.add_argument("--agent", required=True, help="agent id whose corpus to pull (e.g. clawdbot)")
    p_tune_ptp.add_argument(
        "--from-host",
        required=True,
        help="ssh alias to pull corpus from (e.g. neb-server, cliqk-server)",
    )
    p_tune_ptp.add_argument(
        "--base-model", default=None, help="HF model id (default Qwen/Qwen3-8B)"
    )
    p_tune_ptp.add_argument("--rank", type=int, default=None, help="LoRA num-layers (default 16)")
    p_tune_ptp.add_argument("--iters", type=int, default=None, help="training iters (default 600)")
    p_tune_ptp.add_argument(
        "--min-new-examples",
        type=int,
        default=None,
        help="threshold: skip if fewer than this many new examples (default 50)",
    )
    p_tune_ptp.add_argument(
        "--no-push",
        action="store_true",
        help="train locally only; do not scp the adapter back to --from-host",
    )
    p_tune_ptp.add_argument("--verbose", "-v", action="store_true")
    p_tune_ptp.set_defaults(func=cmd_tune_pull_train_push)

    p_tune_train = sub.add_parser(
        "tune-train-local",
        help="LoRA fine-tune locally (auto-detects backend: MLX on macOS arm64, Unsloth on Linux+CUDA)",
    )
    p_tune_train.add_argument("--agent", required=True)
    p_tune_train.add_argument(
        "--backend",
        choices=("auto", "mlx", "unsloth"),
        default="auto",
        help="training backend (default auto: MLX on macOS arm64, Unsloth on Linux+CUDA)",
    )
    p_tune_train.add_argument(
        "--base-model",
        default=None,
        help="HF model id (default Qwen/Qwen3-8B, switchable to meta-llama/Meta-Llama-3.1-8B-Instruct etc.)",
    )
    p_tune_train.add_argument("--rank", type=int, default=None, help="LoRA rank / num-layers (default 16)")
    p_tune_train.add_argument("--epochs", type=int, default=None, help="(Unsloth backend) training epochs (default 3)")
    p_tune_train.add_argument("--iters", type=int, default=None, help="(MLX backend) training iterations (default 600)")
    p_tune_train.add_argument(
        "--min-new-examples",
        type=int,
        default=None,
        help="threshold: skip if fewer than this many new examples since last training (default 50)",
    )
    p_tune_train.add_argument(
        "--export-gguf",
        action="store_true",
        help="(Unsloth backend) after training, export the merged LoRA model to GGUF for Ollama loadability",
    )
    p_tune_train.add_argument("--verbose", "-v", action="store_true")
    p_tune_train.set_defaults(func=cmd_tune_train_local)

    p_tune_status = sub.add_parser(
        "tune-status",
        help="check the most recent local LoRA training run and report the current adapter / GGUF path",
    )
    p_tune_status.add_argument("--agent", required=True)
    p_tune_status.add_argument("--verbose", "-v", action="store_true")
    p_tune_status.set_defaults(func=cmd_tune_status)

    p_reflect = sub.add_parser(
        "reflect",
        help="distill recent agent sessions into curated memory files",
    )
    p_reflect.add_argument(
        "--agent",
        required=True,
        help="openclaw agent id whose sessions to reflect on (e.g. jarvis, clawdbot)",
    )
    p_reflect.add_argument(
        "--host",
        default=None,
        help="hostname tag for provenance (defaults to socket.gethostname())",
    )
    p_reflect.add_argument(
        "--model",
        default=None,
        help="LLM model id (default $HYPERSWARM_REFLECT_MODEL or gpt-4o-mini)",
    )
    p_reflect.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="cap on user/assistant turns sent per session (default 60)",
    )
    p_reflect.add_argument("--verbose", "-v", action="store_true")
    p_reflect.set_defaults(func=cmd_reflect)

    args = parser.parse_args()
    return args.func(args)


def cmd_watch(args: argparse.Namespace) -> int:
    """Long-running watch loop. Blocks until interrupted."""
    from hyperswarm.watchers.openclaw_sessions import OpenClawSessionWatcher

    kwargs: dict = {"agents": list(args.agent)}
    if args.poll_interval is not None:
        kwargs["poll_interval_s"] = args.poll_interval
    if args.debounce is not None:
        kwargs["debounce_s"] = args.debounce
    if args.no_tune:
        kwargs["enable_tune"] = False
    OpenClawSessionWatcher(**kwargs).loop()
    return 0


def cmd_tune_collect(args: argparse.Namespace) -> int:
    """Append new user/assistant pairs to the fine-tune corpus."""
    from hyperswarm.tuners.openclaw_corpus import OpenClawCorpusCollector

    kwargs: dict = {"agent": args.agent}
    if args.host:
        kwargs["host"] = args.host
    result = OpenClawCorpusCollector(**kwargs).run()
    if args.verbose:
        print(json.dumps(result, indent=2))
    else:
        print(
            f"agent={result['agent']} appended={result.get('appended', 0)} "
            f"total_examples={result.get('total_examples', 0)}"
        )
    return 0


def cmd_tune_pull_train_push(args: argparse.Namespace) -> int:
    """End-to-end Mac trainer: scp corpus from a server, train via MLX, scp
    the resulting adapter back. Designed for the workflow 'Mac is primary
    trainer when awake; cloud GPU is fallback when Mac is off.'"""
    import subprocess
    from pathlib import Path

    home = Path.home()
    local_corpus_dir = home / ".openclaw" / "tune" / args.agent
    local_corpus = local_corpus_dir / "corpus.jsonl"
    remote_corpus = f"~/.openclaw/tune/{args.agent}/corpus.jsonl"

    local_corpus_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ptp] pulling {args.from_host}:{remote_corpus} → {local_corpus}", file=sys.stderr)
    pull = subprocess.run(
        ["scp", f"{args.from_host}:{remote_corpus}", str(local_corpus)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if pull.returncode != 0:
        print(f"[ptp] scp pull failed: {pull.stderr.strip()}", file=sys.stderr)
        return 1

    train_args = ["hyperswarm", "tune-train-local", "--agent", args.agent, "--backend", "mlx"]
    if args.base_model:
        train_args += ["--base-model", args.base_model]
    if args.rank is not None:
        train_args += ["--rank", str(args.rank)]
    if args.iters is not None:
        train_args += ["--iters", str(args.iters)]
    if args.min_new_examples is not None:
        train_args += ["--min-new-examples", str(args.min_new_examples)]
    if args.verbose:
        train_args.append("--verbose")
    print(f"[ptp] running: {' '.join(train_args)}", file=sys.stderr)
    train = subprocess.run(train_args, capture_output=False)
    if train.returncode != 0:
        return train.returncode

    if args.no_push:
        print("[ptp] --no-push: skipping adapter upload to source host", file=sys.stderr)
        return 0

    # Find the adapter dir written by the most recent successful run
    from hyperswarm.tuners.lora_mlx import MLXLoRATrainer

    state = MLXLoRATrainer(agent=args.agent).status()
    adapter_path = state.get("current_adapter")
    if not adapter_path:
        print("[ptp] no adapter recorded after training; nothing to push", file=sys.stderr)
        return 1
    adapter_dir = Path(adapter_path)
    if not adapter_dir.exists():
        print(f"[ptp] state references missing adapter dir {adapter_dir}", file=sys.stderr)
        return 1

    # Push the adapter dir + the state file (so the source host knows about
    # the new adapter)
    run_dir = adapter_dir.parent
    relative = run_dir.relative_to(home)
    remote_run_dir = f"~/{relative}"
    print(f"[ptp] pushing {run_dir} → {args.from_host}:{remote_run_dir}", file=sys.stderr)
    subprocess.run(
        ["ssh", args.from_host, f"mkdir -p {remote_run_dir}"],
        check=True,
        timeout=60,
    )
    push = subprocess.run(
        ["scp", "-r", str(run_dir) + "/", f"{args.from_host}:{remote_run_dir}/"],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if push.returncode != 0:
        print(f"[ptp] scp push failed: {push.stderr.strip()}", file=sys.stderr)
        return 1

    state_file = home / ".local" / "state" / "hyperswarm" / "tune" / args.agent / "finetune-state.json"
    if state_file.exists():
        remote_state = f".local/state/hyperswarm/tune/{args.agent}/finetune-state.json"
        subprocess.run(
            ["ssh", args.from_host, f"mkdir -p $(dirname {remote_state})"],
            check=True,
            timeout=60,
        )
        subprocess.run(
            ["scp", str(state_file), f"{args.from_host}:{remote_state}"],
            check=True,
            timeout=60,
        )
        print(f"[ptp] pushed state file → {args.from_host}:{remote_state}", file=sys.stderr)

    print(
        f"[ptp] done. agent={args.agent} adapter={adapter_path} pushed-to={args.from_host}",
        file=sys.stderr,
    )
    return 0


def _resolve_train_backend(requested: str) -> str:
    """Map --backend to a concrete backend name. 'auto' picks MLX on macOS
    arm64 if mlx-lm is importable, else Unsloth on a CUDA host, else raises."""
    if requested != "auto":
        return requested
    from hyperswarm.tuners.lora_mlx import is_mlx_available
    if is_mlx_available():
        return "mlx"
    from hyperswarm.tuners.lora_local import is_cuda_available
    if is_cuda_available():
        return "unsloth"
    raise SystemExit(
        "tune-train-local: no compatible backend detected. Install mlx-lm on "
        "macOS arm64 or torch+unsloth on a CUDA host, then re-run with "
        "--backend mlx or --backend unsloth explicitly."
    )


def cmd_tune_train_local(args: argparse.Namespace) -> int:
    """LoRA fine-tune locally. Auto-detects MLX (Mac primary) vs Unsloth (CUDA)."""
    backend = _resolve_train_backend(args.backend)
    kwargs: dict = {"agent": args.agent}
    if args.base_model:
        kwargs["base_model"] = args.base_model
    if args.min_new_examples is not None:
        kwargs["min_new_examples"] = args.min_new_examples
    if backend == "mlx":
        from hyperswarm.tuners.lora_mlx import MLXLoRATrainer
        if args.rank is not None:
            kwargs["num_layers"] = args.rank
        if args.iters is not None:
            kwargs["iters"] = args.iters
        if args.export_gguf:
            print(
                "warning: --export-gguf is Unsloth-only (deferred for MLX). "
                "Ignoring; the MLX adapter is loadable via mlx_lm.generate.",
                file=sys.stderr,
            )
        trainer = MLXLoRATrainer(**kwargs)
    else:  # unsloth
        from hyperswarm.tuners.lora_local import LocalLoRATrainer
        if args.rank is not None:
            kwargs["lora_rank"] = args.rank
        if args.epochs is not None:
            kwargs["n_epochs"] = args.epochs
        kwargs["export_gguf"] = args.export_gguf
        trainer = LocalLoRATrainer(**kwargs)

    result = trainer.train()
    result["backend"] = backend
    if args.verbose:
        print(json.dumps(result, indent=2))
    else:
        st = result.get("status")
        if st == "completed":
            print(
                f"agent={args.agent} backend={backend} status=completed "
                f"adapter={result.get('adapter_path')} "
                f"gguf={result.get('gguf_path') or '(none)'}"
            )
        else:
            print(
                f"agent={args.agent} backend={backend} status={st} "
                f"reason={result.get('reason', '')}"
            )
    return 0 if result.get("status") in ("completed", "skipped") else 1


def cmd_tune_status(args: argparse.Namespace) -> int:
    """Read the local-LoRA state and report the most recent adapter / GGUF path.
    Backend-agnostic — both lora_local.py and lora_mlx.py write the same
    state-file shape, so a single status reader works for either."""
    from hyperswarm.tuners.lora_mlx import MLXLoRATrainer

    # Use the MLX class purely for its state-loading logic — both backends
    # share the same on-disk schema so this works regardless of which one
    # actually trained the most recent run. The `backend` field in state
    # tells us which trainer wrote it.
    result = MLXLoRATrainer(agent=args.agent).status()
    if args.verbose:
        print(json.dumps(result, indent=2))
    else:
        print(
            f"agent={args.agent} backend={result.get('backend')} "
            f"last_run={result.get('last_run_status')} "
            f"current_adapter={result.get('current_adapter')} "
            f"current_gguf={result.get('current_gguf')}"
        )
    return 0


def cmd_reflect(args: argparse.Namespace) -> int:
    """Run a single reflection pass for one agent."""
    from hyperswarm.reflectors.openclaw_session import OpenClawSessionReflector

    kwargs: dict = {"agent": args.agent}
    if args.host:
        kwargs["host"] = args.host
    if args.model:
        kwargs["model"] = args.model
    if args.max_turns is not None:
        kwargs["max_turns"] = args.max_turns
    result = OpenClawSessionReflector(**kwargs).run()
    if args.verbose:
        print(json.dumps(result, indent=2))
    else:
        print(
            f"agent={result['agent']} sessions_processed={result.get('sessions_processed', 0)} "
            f"written={result.get('written', 0)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
