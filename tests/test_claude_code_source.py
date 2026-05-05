"""ClaudeCodeSource contract tests:
  - install() is idempotent and merges into existing settings without nuking
  - install() updates an existing HyperSwarm hook in place
  - capture() handles missing transcript gracefully
  - capture() reads last user + last assistant + files touched from JSONL
  - capture() never raises, even on malformed input
  - uninstall() is a no-op when nothing was installed
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from hyperswarm.sources.claude_code import ClaudeCodeSource


def test_install_idempotent(tmp_path: Path):
    settings = tmp_path / "settings.json"
    src = ClaudeCodeSource({"settings_path": str(settings)})
    src.install()
    src.install()
    src.install()

    data = json.loads(settings.read_text())
    stop = data["hooks"]["Stop"]
    cmds = [h["command"] for entry in stop for h in entry["hooks"]]
    hyperswarm_cmds = [c for c in cmds if "capture --runtime claude-code" in c]
    assert len(hyperswarm_cmds) == 1, f"expected 1 hook, got {len(hyperswarm_cmds)}"


def test_install_preserves_other_hooks(tmp_path: Path):
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({
        "theme": "dark",
        "hooks": {
            "PreToolUse": [{"matcher": "X", "hooks": [{"type": "command", "command": "echo pre"}]}],
            "Stop": [{"matcher": "Y", "hooks": [{"type": "command", "command": "echo stop"}]}],
        },
    }))
    src = ClaudeCodeSource({"settings_path": str(settings)})
    src.install()

    data = json.loads(settings.read_text())
    assert data["theme"] == "dark"
    assert data["hooks"]["PreToolUse"][0]["hooks"][0]["command"] == "echo pre"
    stop_cmds = [h["command"] for entry in data["hooks"]["Stop"] for h in entry["hooks"]]
    assert "echo stop" in stop_cmds
    assert any("capture --runtime claude-code" in c for c in stop_cmds)


def test_install_updates_existing_hyperswarm_hook(tmp_path: Path):
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({
        "hooks": {
            "Stop": [{
                "matcher": ".*",
                "hooks": [{"type": "command", "command": "hyperswarm capture --runtime claude-code # OLD"}],
            }],
        },
    }))
    src = ClaudeCodeSource({
        "settings_path": str(settings),
        "hook_command": "hyperswarm capture --runtime claude-code || true",
    })
    src.install()

    data = json.loads(settings.read_text())
    cmds = [h["command"] for entry in data["hooks"]["Stop"] for h in entry["hooks"]]
    hyperswarm_cmds = [c for c in cmds if "capture --runtime claude-code" in c]
    assert len(hyperswarm_cmds) == 1
    assert "OLD" not in hyperswarm_cmds[0]
    assert "|| true" in hyperswarm_cmds[0]


def test_uninstall_removes_only_our_hook(tmp_path: Path):
    settings = tmp_path / "settings.json"
    src = ClaudeCodeSource({"settings_path": str(settings)})
    src.install()
    # Add an unrelated Stop hook alongside ours.
    data = json.loads(settings.read_text())
    data["hooks"]["Stop"].append({"matcher": "Z", "hooks": [{"type": "command", "command": "echo other"}]})
    settings.write_text(json.dumps(data))

    src.uninstall()

    data = json.loads(settings.read_text())
    cmds = [h["command"] for entry in data["hooks"].get("Stop", []) for h in entry["hooks"]]
    assert any("echo other" in c for c in cmds)
    assert not any("capture --runtime claude-code" in c for c in cmds)


def test_uninstall_no_op_when_nothing_present(tmp_path: Path):
    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    src = ClaudeCodeSource({"settings_path": str(settings)})
    src.uninstall()  # must not raise


def test_default_hook_command_uses_absolute_path(tmp_path: Path, monkeypatch):
    """The hook command must resolve to an absolute hyperswarm binary at
    install time so it works regardless of how Claude Code is launched (Dock
    vs terminal — different PATH)."""
    fake_bin = tmp_path / "fake-hyperswarm-bin"
    fake_bin.write_text("#!/bin/sh\nexit 0\n")
    fake_bin.chmod(0o755)

    # shutil.which honours PATH, so put fake_bin's dir at the front.
    monkeypatch.setenv("PATH", f"{fake_bin.parent}:{os.environ.get('PATH', '')}")
    # rename so shutil.which("hyperswarm") finds it
    target = fake_bin.parent / "hyperswarm"
    fake_bin.rename(target)

    settings = tmp_path / "settings.json"
    src = ClaudeCodeSource({"settings_path": str(settings)})
    src.install()

    data = json.loads(settings.read_text())
    cmds = [h["command"] for entry in data["hooks"]["Stop"] for h in entry["hooks"]]
    assert any(str(target) in c for c in cmds), f"expected absolute path {target} in hook command, got {cmds}"


def test_user_hook_command_override_wins(tmp_path: Path):
    """If the user provides hook_command in config.toml, install() respects
    it — does not blow away their override with the resolved absolute path."""
    settings = tmp_path / "settings.json"
    src = ClaudeCodeSource({
        "settings_path": str(settings),
        "hook_command": "my-custom-command --runtime claude-code",
    })
    src.install()
    cmd = json.loads(settings.read_text())["hooks"]["Stop"][0]["hooks"][0]["command"]
    assert cmd == "my-custom-command --runtime claude-code"


def test_capture_with_missing_transcript_returns_entry(tmp_path: Path):
    src = ClaudeCodeSource({"settings_path": str(tmp_path / "settings.json")})
    entry = src.capture({"cwd": "/tmp", "session_id": "sid", "transcript_path": "/no/such/file"})
    assert entry.runtime == "claude-code"
    assert entry.cwd == "/tmp"
    assert entry.session_id == "sid"
    assert entry.body  # non-empty fallback


def test_capture_extracts_last_turns_from_jsonl(tmp_path: Path):
    transcript = tmp_path / "session.jsonl"
    rows = [
        {"role": "user", "message": {"content": "older user msg"}},
        {"role": "assistant", "message": {"content": "older assistant"}},
        {"role": "user", "message": {"content": "What's the next step?"}},
        {
            "role": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "We should refactor X."},
                    {"type": "tool_use", "name": "Edit", "input": {"file_path": "/a/b.py"}},
                    {"type": "tool_use", "name": "Write", "input": {"file_path": "/a/c.py"}},
                ],
            },
        },
    ]
    transcript.write_text("\n".join(json.dumps(r) for r in rows))

    src = ClaudeCodeSource({"settings_path": str(tmp_path / "settings.json")})
    entry = src.capture({
        "cwd": "/work",
        "session_id": "sid",
        "transcript_path": str(transcript),
    })
    assert "What's the next step?" in entry.body
    assert "We should refactor X" in entry.body
    assert "/a/b.py" in entry.body
    assert "/a/c.py" in entry.body
    assert entry.summary == "What's the next step?"


def test_capture_never_raises_on_garbage_transcript(tmp_path: Path):
    transcript = tmp_path / "broken.jsonl"
    transcript.write_text("this is not json\n{also broken\n")
    src = ClaudeCodeSource({"settings_path": str(tmp_path / "settings.json")})
    # Must not raise
    entry = src.capture({
        "cwd": "/work", "session_id": "sid", "transcript_path": str(transcript)
    })
    assert entry.runtime == "claude-code"
