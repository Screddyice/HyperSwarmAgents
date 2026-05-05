from hyperswarm.sources.claude_code import ClaudeCodeSource
from hyperswarm.sources.codex import CodexSource
from hyperswarm.sources.openclaw import OpenClawSource

SOURCE_REGISTRY: dict[str, type] = {
    "claude_code": ClaudeCodeSource,
    "claude-code": ClaudeCodeSource,
    "codex": CodexSource,
    "openclaw": OpenClawSource,
}

__all__ = ["ClaudeCodeSource", "CodexSource", "OpenClawSource", "SOURCE_REGISTRY"]
