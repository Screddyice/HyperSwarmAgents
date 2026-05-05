from hyperswarm.sources.claude_code import ClaudeCodeSource
from hyperswarm.sources.codex import CodexSource
from hyperswarm.sources.openclaw import OpenClawSource
from hyperswarm.sources.openclaw_runs import OpenClawRunsSource

SOURCE_REGISTRY: dict[str, type] = {
    "claude_code": ClaudeCodeSource,
    "claude-code": ClaudeCodeSource,
    "codex": CodexSource,
    "openclaw": OpenClawSource,
    "openclaw_runs": OpenClawRunsSource,
}

__all__ = [
    "ClaudeCodeSource",
    "CodexSource",
    "OpenClawSource",
    "OpenClawRunsSource",
    "SOURCE_REGISTRY",
]
