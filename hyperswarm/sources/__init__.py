from hyperswarm.sources.claude_code import ClaudeCodeSource
from hyperswarm.sources.claude_harness import ClaudeHarnessSource
from hyperswarm.sources.claude_mem_session import ClaudeMemSessionSource
from hyperswarm.sources.codex import CodexSource
from hyperswarm.sources.openclaw import OpenClawSource
from hyperswarm.sources.openclaw_runs import OpenClawRunsSource

SOURCE_REGISTRY: dict[str, type] = {
    "claude_code": ClaudeCodeSource,
    "claude-code": ClaudeCodeSource,
    "claude_harness": ClaudeHarnessSource,
    "claude-harness": ClaudeHarnessSource,
    "claude_mem_session": ClaudeMemSessionSource,
    "claude-mem-session": ClaudeMemSessionSource,
    "codex": CodexSource,
    "openclaw": OpenClawSource,
    "openclaw_runs": OpenClawRunsSource,
}

__all__ = [
    "ClaudeCodeSource",
    "ClaudeHarnessSource",
    "ClaudeMemSessionSource",
    "CodexSource",
    "OpenClawSource",
    "OpenClawRunsSource",
    "SOURCE_REGISTRY",
]
