from hyperswarm.sources.claude_code import ClaudeCodeSource
from hyperswarm.sources.claude_harness import ClaudeHarnessSource
from hyperswarm.sources.claude_mem_session import ClaudeMemSessionSource
from hyperswarm.sources.codex import CodexSource
from hyperswarm.sources.mem0_session import Mem0SessionSource

# retired 2026-06-08: openclaw decommissioned, replaced by Hermes memory provider
# claude_mem_session kept for historical capture/backfill; live config uses mem0_session

SOURCE_REGISTRY: dict[str, type] = {
    "claude_code": ClaudeCodeSource,
    "claude-code": ClaudeCodeSource,
    "claude_harness": ClaudeHarnessSource,
    "claude-harness": ClaudeHarnessSource,
    "claude_mem_session": ClaudeMemSessionSource,
    "claude-mem-session": ClaudeMemSessionSource,
    "mem0_session": Mem0SessionSource,
    "mem0-session": Mem0SessionSource,
    "codex": CodexSource,
}

__all__ = [
    "ClaudeCodeSource",
    "ClaudeHarnessSource",
    "ClaudeMemSessionSource",
    "Mem0SessionSource",
    "CodexSource",
    "SOURCE_REGISTRY",
]
