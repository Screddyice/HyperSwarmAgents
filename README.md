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
project: hyper_flow     # optional, set by Scope plugin (e.g. git_remote → repo name)
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

## Reflectors — making the brain smarter over time

Sources capture-and-store. **Reflectors synthesize across sessions** so the agent gets smarter the longer it's running. Pattern reference: Park et al, "Generative Agents: Interactive Simulacra of Human Behavior" (2023) — Memory Stream → Reflection → Retrieval → Planning. This module implements the Reflection layer.

### `hyperswarm reflect`

```bash
hyperswarm reflect --agent jarvis
```

Reads new turns from `~/.openclaw/agents/<agent>/sessions/*.jsonl`, calls an LLM with a strict "extract only high-signal learnings" prompt, and writes zero-or-more YAML-frontmatter markdown blocks into `~/.openclaw/claude-code-history/projects/-Users-screddy-projects/memory/server-learned/<agent>/`. Per-session cursor in `~/.local/state/hyperswarm/reflect/<agent>.json` keeps the next run idempotent.

The output dir is the same one openclaw's `memory_search` already indexes via `extraPaths`, so distilled memories surface to the agent on the next reindex tick. If you also sync the dir to other servers/laptops (via the existing rsync sync stack or your own pipe), reflections written on one host become available to all of them.

**Cost**: with `gpt-4o-mini` (default) one reflection run is roughly $0.001-0.005 per session depending on turn count. Run hourly per agent, expect ~$1/month/agent. Override with `--model` or `HYPERSWARM_REFLECT_MODEL` env var.

**Auth**: the reflection LLM call uses `OPENAI_API_KEY` from the process environment.

**Recommended cron** on each server:

```bash
0 */6 * * * /home/ubuntu/.local/bin/hyperswarm reflect --agent jarvis >> ~/.local/state/hyperswarm/reflect.log 2>&1
```

### Building your own reflector

```python
from hyperswarm.reflectors.openclaw_session import OpenClawSessionReflector

result = OpenClawSessionReflector(
    agent="jarvis",
    host="my-server",
    output_base="~/wherever/memory/server-learned",
    llm_call=my_llm_function,  # for tests or alternate providers
).run()
```

The `llm_call` is fully injectable (any `messages: list[dict] -> str` callable), so swapping providers (Anthropic, local, Ollama) is straightforward.

## Tuners — slow learning. Update model weights, not just context.

Where Reflectors update *context* (fast learning), Tuners update *weights* (slow learning). Karpathy's framing: there are two learning channels in any AI system, and a personal assistant that gets genuinely smarter from your interactions needs both.

### Pipeline

```
session JSONLs                 (raw)
       │
       ▼ hyperswarm tune-collect --agent <id>
       │
       ├── pair user→assistant turns
       ├── filter (length bounds, drop tool-result noise)
       └── append OpenAI-chat-format examples to corpus.jsonl
       │
       ▼ hyperswarm tune-train-local --agent <id>     (run on a CUDA host)
       │
       ├── guard: skip if a previous run still claims "running"
       ├── guard: skip if new examples since last run < threshold (default 50)
       └── load corpus → Unsloth LoRA train → save adapter (+ optional GGUF)
       │
       ▼ hyperswarm tune-status --agent <id>
       │
       └── read state, report current adapter / GGUF path
                  │
                  ▼
        ~/.openclaw/tune/<agent>/lora-output/<run-id>/adapter/
        + (optional) Qwen3-8B-q4_k_m.gguf  ← Ollama can load directly
            now available for selective routing
```

### Backend: self-hosted LoRA, two paths

Earlier iterations used hosted fine-tune services (OpenAI, Together). OpenAI sunset their endpoint and we ruled out hosted services entirely. Current model: own model, own data, own training, downloadable weights. Aligned with Karpathy's "weights vs context" framing — this is the weights side.

Two backends ship with the same CLI shape; `tune-train-local` auto-picks based on host:

| Backend | When picked | Why |
|---|---|---|
| **MLX** (`lora_mlx.py`) — primary | macOS arm64 (M-series) | Free, private, fast on Apple Silicon Max-tier (unified memory; no GPU partition limits). Wraps `mlx_lm.lora --train`. |
| **Unsloth** (`lora_local.py`) — secondary | Linux + CUDA | Fallback when the Mac is offline. Wraps Unsloth + TRL. |

Default base model: `Qwen/Qwen3-8B`. Switchable via `--base-model` to any HF model id either backend supports.

```bash
# Auto-detect (MLX on Mac, Unsloth on CUDA, error otherwise):
hyperswarm tune-train-local --agent clawdbot

# Force a specific backend:
hyperswarm tune-train-local --agent clawdbot --backend mlx
hyperswarm tune-train-local --agent clawdbot --backend unsloth
```

### Mac as primary trainer: `tune-pull-train-push`

The full Mac-side workflow in one command — pulls a server's accumulated corpus, trains locally with MLX, pushes the resulting adapter back:

```bash
hyperswarm tune-pull-train-push --agent clawdbot --from-host neb-server
```

Sequence: `scp neb-server:~/.openclaw/tune/clawdbot/corpus.jsonl ~/.openclaw/tune/clawdbot/corpus.jsonl` → `hyperswarm tune-train-local --agent clawdbot --backend mlx` → wait for MLX to train against the corpus on Apple Silicon GPU → on success, `scp -r` the adapter directory + state file back to `neb-server`. End-to-end in one command, idempotent, threshold-gated.

Pass `--no-push` to keep the adapter local-only (useful for testing).

### When the Mac is off: server-side fallback

If your Mac isn't reachable when you want to train, the same corpus can be trained on a Linux+CUDA host directly. SSH into the GPU box, pull the corpus, run `hyperswarm tune-train-local --agent <id>` — the auto-detector picks Unsloth and the run produces adapter + optional GGUF export. State files share the same shape across backends, so a Mac-trained adapter and a CUDA-trained adapter are interchangeable downstream.

State per agent:

- Corpus:        `~/.openclaw/tune/<agent>/corpus.jsonl`
- Cursors:       `~/.local/state/hyperswarm/tune/<agent>/corpus-cursors.json`
- Training state: `~/.local/state/hyperswarm/tune/<agent>/finetune-state.json` (`backend: "lora-local"`, `current_adapter`, `current_gguf`, full history)

The state file shape is intentionally backend-agnostic — the `current_adapter` / `current_gguf` keys let any router downstream understand "which model represents this agent right now" regardless of which trainer wrote it.

### Cost

- Cloud GPU ad-hoc (RunPod 4090): $0.30-0.50/hr × ~30 min per cycle ≈ $0.15-0.25/cycle
- Owned 4090: one-time hardware + ~$0.05 of electricity per cycle
- Inference: free (run on your existing CPU server via Ollama, or on the same GPU host)
- vs hosted fine-tune ($5-15/month/agent and dependent on a vendor that may sunset): substantially cheaper at scale and zero vendor lock.

### Dependencies

```bash
pip install unsloth trl datasets torch
```

CUDA-only. Cleanly raises `RuntimeError("Local LoRA training requires ...")` on a CPU host so the orchestration code can be imported and tested anywhere — only the `_real_train` path needs the GPU.

### Optional: GGUF export for Ollama

Pass `--export-gguf` and the trainer also writes a quantized GGUF file alongside the adapter. Ollama can `ollama create my-jarvis -f Modelfile` against that GGUF, making the personalized model loadable on any of the CPU inference servers.

## Watchers — event-driven, no constant crons

Reflectors and Tuners are CLI commands. Running them on a calendar (`*/6 * * *`) wastes both LLM dollars and your patience. **Watchers** are long-running daemons that fire reflect+tune ONLY when a session has actually ended (idle for N seconds).

### `hyperswarm watch`

```bash
hyperswarm watch --agent jarvis --agent clawdbot
```

Polls session JSONLs every 30 seconds (configurable). When a session has been idle for 5 minutes (configurable via `--debounce`), the watcher fires:

1. `hyperswarm reflect --agent <id>`
2. `hyperswarm tune-collect --agent <id>`

Both are idempotent (cursors + thresholds), so re-firing on a session that didn't add new turns is free. Use `--no-tune` to run reflect-only.

The watcher does **not** auto-fire `tune-train-local` — training requires a CUDA GPU which the watcher's host (typically a CPU inference server) lacks. Training is a manual step run on a separate GPU host that pulls the corpus, trains, and writes the adapter back. This intentional split keeps the watcher cheap and reliable.

### Systemd unit (recommended)

Drop one unit per server, watching all agents on that host:

```ini
# ~/.config/systemd/user/hyperswarm-watch.service
[Unit]
Description=HyperSwarm session watcher (event-driven reflect + tune)
After=network-online.target

[Service]
EnvironmentFile=-/home/ubuntu/.openclaw/.env
ExecStart=%h/.local/bin/hyperswarm watch --agent jarvis --agent clawdbot
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
```

Then `systemctl --user enable --now hyperswarm-watch.service`. The watcher uses ~1% CPU at idle and never spawns LLM calls itself; all costs are gated through the reflect/tune subcommands' own thresholds.

## Status

This is Phase 1 of an open-source rollout: scaffolding, plugin interfaces, reference Scope + Store implementations, tests. Phase 2 fills in the Source reference implementations (Claude Code Stop hook, Codex wrapper, OpenClaw watcher); Phase 3 adds the read-on-start digest that surfaces recent context to a new session on any runtime. Reflectors (Phase 4) synthesize across sessions for context-layer compounding. Tuners (Phase 5, this PR) close the loop by updating model weights from accumulated interactions. Watchers (Phase 5) replace crons with event-driven triggers.

## License

MIT
