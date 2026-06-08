from hyperswarm.sources.claude_code import ClaudeCodeSource
from hyperswarm.sources.claude_harness import ClaudeHarnessSource
from hyperswarm.sources.claude_mem_session import ClaudeMemSessionSource
from hyperswarm.sources.codex import CodexSource

# retired 2026-06-08: openclaw decommissioned, replaced by Hermes memory provider

SOURCE_REGISTRY: dict[str, type] = {
    "claude_code": ClaudeCodeSource,
    "claude-code": ClaudeCodeSource,
    "claude_harness": ClaudeHarnessSource,
    "claude-harness": ClaudeHarnessSource,
    "claude_mem_session": ClaudeMemSessionSource,
    "claude-mem-session": ClaudeMemSessionSource,
    "codex": CodexSource,
}

__all__ = [
    "ClaudeCodeSource",
    "ClaudeHarnessSource",
    "ClaudeMemSessionSource",
    "CodexSource",
    "SOURCE_REGISTRY",
]
