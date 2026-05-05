"""CodexSource — capture OpenAI Codex CLI sessions via a shell wrapper.

How it wires up:

  1. `install()` writes a wrapper script at <wrapper_path> (default
     ~/.local/bin/codex) that:
       a. Records a start timestamp.
       b. Invokes the real codex binary with all args.
       c. On exit (success or failure), runs `hyperswarm capture --runtime
          codex --since-ts <start>` so HyperSwarm scans ~/.codex/log/ for
          files modified during the session and snapshots them.

     The wrapper uses `|| true` on the capture step so a HyperSwarm bug never
     surfaces as a non-zero exit code from `codex`.

  2. The wrapper assumes ~/.local/bin precedes /opt/homebrew/bin in $PATH —
     i.e. it shadows the brew binary. `install()` validates this and refuses
     to write a wrapper that wouldn't actually intercept invocations.

  3. `capture()` reads files in ~/.codex/log/ modified after `since_ts` and
     produces an Entry summarising the session. Codex's own log format is
     opaque text — we don't pretend to parse it; we just attach the most
     recent log file's last ~3 KB as the entry body.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from hyperswarm.core.entry import Entry
from hyperswarm.core.source import Source

DEFAULT_REAL_BINARY = "/opt/homebrew/bin/codex"
DEFAULT_WRAPPER_PATH = "~/.local/bin/codex"
DEFAULT_LOG_DIR = "~/.codex/log"
WRAPPER_SENTINEL = "# managed-by: hyperswarm-agents"


def _resolve_hyperswarm_binary() -> str:
    """Absolute path to hyperswarm at install time, baked into the wrapper.
    Falls back to bare name (relies on PATH at codex run time)."""
    return shutil.which("hyperswarm") or "hyperswarm"


class CodexSource(Source):
    name = "codex"

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self.real_binary = self.config.get("binary", DEFAULT_REAL_BINARY)
        self.wrapper_path = Path(
            os.path.expanduser(self.config.get("wrapper_path", DEFAULT_WRAPPER_PATH))
        )
        self.log_dir = Path(os.path.expanduser(self.config.get("log_dir", DEFAULT_LOG_DIR)))

    # ------------------------------------------------------------- install
    def install(self) -> None:
        """Idempotent: writes the wrapper script if missing or out of date."""
        if not Path(self.real_binary).exists():
            raise RuntimeError(
                f"real codex binary not found at {self.real_binary}; "
                "set 'binary' in source config or `brew install codex`"
            )
        # Ensure ~/.local/bin precedes the real binary's dir on PATH so the
        # wrapper actually intercepts. We can't change the user's shell rc
        # from here, so fail loudly if shadowing won't work.
        path_dirs = os.environ.get("PATH", "").split(":")
        wrapper_dir = str(self.wrapper_path.parent)
        real_dir = str(Path(self.real_binary).parent)
        if wrapper_dir in path_dirs and real_dir in path_dirs:
            if path_dirs.index(wrapper_dir) > path_dirs.index(real_dir):
                raise RuntimeError(
                    f"{wrapper_dir} comes after {real_dir} in PATH; "
                    "wrapper would not intercept. Reorder your shell rc first."
                )

        self.wrapper_path.parent.mkdir(parents=True, exist_ok=True)
        self.wrapper_path.write_text(self._wrapper_script())
        self.wrapper_path.chmod(0o755)

    def uninstall(self) -> None:
        """Remove the wrapper iff it's the one we wrote (sentinel match)."""
        if not self.wrapper_path.exists():
            return
        try:
            content = self.wrapper_path.read_text()
        except Exception:
            return
        if WRAPPER_SENTINEL in content:
            self.wrapper_path.unlink()

    def _wrapper_script(self) -> str:
        hs_bin = _resolve_hyperswarm_binary()
        return f"""#!/usr/bin/env bash
{WRAPPER_SENTINEL}
# Wrapper that intercepts `codex` invocations and snapshots the session
# to HyperSwarm on exit. Auto-generated — do not edit by hand. To regenerate
# run `hyperswarm install --runtime codex`.

set +e
START_TS=$(date +%s)
SESSION_CWD="$PWD"

"{self.real_binary}" "$@"
RC=$?

# Absolute path resolved at install time, so this works even if the user's
# PATH changes after install.
HYPERSWARM_BIN="{hs_bin}"
if [ -x "$HYPERSWARM_BIN" ]; then
    printf '{{"cwd":"%s","since_ts":%s}}' "$SESSION_CWD" "$START_TS" \\
        | "$HYPERSWARM_BIN" capture --runtime codex >/dev/null 2>&1 || true
fi

exit $RC
"""

    # ------------------------------------------------------------- capture
    def capture(self, raw: dict) -> Entry:
        """raw expects: {"cwd": str, "since_ts": int (epoch seconds)}.

        Tolerant of missing fields. since_ts defaults to "5 minutes ago" so a
        manual `hyperswarm capture --runtime codex` invocation still produces
        something useful for debugging.
        """
        import time
        cwd = raw.get("cwd", "") or os.getcwd()
        since_ts = int(raw.get("since_ts") or (time.time() - 300))

        log_files = self._recent_log_files(since_ts)
        summary, body = self._summarise(log_files)
        return Entry(
            runtime=self.name,
            cwd=cwd,
            summary=summary,
            body=body,
            session_id="",
        )

    def _recent_log_files(self, since_ts: int) -> list[Path]:
        if not self.log_dir.exists():
            return []
        out: list[Path] = []
        for f in self.log_dir.iterdir():
            if not f.is_file():
                continue
            try:
                if f.stat().st_mtime >= since_ts:
                    out.append(f)
            except OSError:
                continue
        return sorted(out, key=lambda p: p.stat().st_mtime, reverse=True)

    @staticmethod
    def _summarise(files: list[Path]) -> tuple[str, str]:
        if not files:
            return "Codex session (no log activity)", "(no codex log files modified during session)"
        # Use the most recent log as the body source — Codex log format is
        # opaque, so we just attach the tail and let humans read it.
        latest = files[0]
        try:
            text = latest.read_text(errors="replace")
        except Exception:
            text = ""
        tail = text[-3000:] if len(text) > 3000 else text
        summary = f"Codex session ({latest.name}, {len(files)} log file{'s' if len(files) != 1 else ''})"
        body = f"## Latest log\n\n`{latest.name}`\n\n```\n{tail.strip()}\n```"
        return summary, body
