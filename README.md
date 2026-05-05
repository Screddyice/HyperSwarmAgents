# HyperSwarmAgents

A pluggable orchestration layer for **agentic context sync** — auto-capture decisions and state from one agent runtime, surface them in another.

Different agent runtimes are good at different things (Claude Code for interactive design, Codex CLI for second-opinion review, OpenClaw / Aider / Cursor / Bedrock agents for autonomous execution), but their memories don't talk. You decide something in one runtime, then a different agent re-relitigates the same decision an hour later because it never saw the context.

HyperSwarmAgents fixes that without locking you into any particular runtime, storage backend, or transport. Everything is a swappable plugin.

## How it's structured

Four extension points, each with a tiny interface. You combine reference implementations (or write your own) to match your fleet:

| Extension point | What it does | Reference implementations |
|---|---|---|
| **Source** | Captures session state from a specific runtime — usually via a hook, wrapper, or directory watcher | `claude_code`, `codex`, `openclaw`, `directory_watcher` |
| **Store** | Persists captured entries — append-only by default for auditability | `markdown` (default), `sqlite` |
| **Sync** | Moves entries between nodes when capture happens off the canonical host | `rsync_ssh` (default), `s3`, `git` |
| **Scope** | Tags each entry so reads can filter by project / team / company / whatever | `path_prefix`, `git_remote`, `custom_callable` |

Your `config.toml` picks one of each (or many of `Source` / `Sync`). Adding a new runtime means writing one Source plugin, ~50 lines.

## What gets captured

Each entry is an append-only markdown file with frontmatter:

```markdown
---
runtime: claude-code
scope: NEB              # provided by your Scope plugin
cwd: /Users/you/projects/teamnebula.ai/api
session_id: abc123
timestamp: 2026-05-04T22:14:00Z
summary: One-line digest used by `recent`
---

# Decisions

- ...

# Files touched

- ...

# Blockers / open threads

- ...
```

Markdown was chosen over SQLite for the default store because: (a) `grep` works as a fallback CLI, (b) human-inspectable, (c) syncs cleanly over rsync/git/S3 without schema migrations. Swap in `sqlite` if you want indexed reads at scale.

## Quick start

```bash
pip install hyperswarm-agents

# Point at your config (defaults to ~/.config/hyperswarm/config.toml)
hyperswarm install --runtime claude-code   # writes the Stop hook
hyperswarm install --runtime codex         # writes the codex wrapper

# Auto-capture is now running. Read from any node:
hyperswarm recent --since 24h
hyperswarm recent --scope NEB
hyperswarm recent --runtime codex
```

## Reference setup: four-instance fleet

The canonical example (in `examples/four-instances/`) wires four runtimes — local Claude Code, local Codex CLI, plus two remote OpenClaw nodes — into one shared memory:

```
                  ┌────────────────────┐
   Claude Code ───►   markdown store    ◄─── Codex CLI
   (local hook)    │  ~/HyperSwarm/...   │   (cli wrapper)
                  └─────────▲─▲─────────┘
                            │ │ rsync over SSH
                  ┌─────────┘ └─────────┐
            OpenClaw A             OpenClaw B
            (remote)               (remote)
```

Each node only writes its own entries; the central host pulls from remote nodes via the rsync-ssh sync plugin. Path-based scope tags every entry so cross-runtime reads stay project-scoped.

See `examples/four-instances/config.toml` for the full wiring.

## Writing a Source plugin

Roughly:

```python
from hyperswarm.core import Source, Entry

class CursorSource(Source):
    name = "cursor"

    def install(self) -> None:
        # Write your hook / wrapper / watcher
        ...

    def capture(self, raw: dict) -> Entry:
        # Convert raw runtime output into an Entry
        return Entry(
            runtime=self.name,
            cwd=raw["cwd"],
            session_id=raw["session_id"],
            summary=...,
            body=...,
        )
```

Register in your `config.toml`:

```toml
[[source]]
type = "cursor"
```

That's it. The orchestrator calls `install()` on first run and `capture()` on every hook event.

## Status

This is Phase 1 of an open-source rollout: scaffolding, plugin interfaces, reference Scope + Store implementations, tests. Phase 2 fills in the Source reference implementations (Claude Code Stop hook, Codex wrapper, OpenClaw watcher); Phase 3 adds the read-on-start digest that surfaces recent context to a new session on any runtime.

## License

MIT
