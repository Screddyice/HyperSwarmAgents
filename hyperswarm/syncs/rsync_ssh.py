"""RsyncSshSync — push/pull entries directories over SSH using rsync.

Why rsync over `cp` or shipping our own diff: rsync only transfers changed
files, handles partial failures gracefully, and is universally available.
The only real prerequisite is SSH key auth to the remote.

Config:

    [[sync]]
    type = "rsync_ssh"
    direction = "push"        # or "pull"
    to_host = "shawn-mac"
    to_path = "~/HyperSwarm/entries"
    from_path = "~/HyperSwarm/entries"   # optional, defaults to to_path
    only_on_host = "neb-server"          # optional gate — sync only runs
                                         # when local hostname matches
"""
from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

from hyperswarm.core.host import get_host_identity
from hyperswarm.core.sync import Sync


class RsyncSshSync(Sync):
    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self.direction = self.config.get("direction", "push")
        if self.direction not in ("push", "pull"):
            raise ValueError(f"direction must be 'push' or 'pull', got {self.direction!r}")
        self.to_host = self.config.get("to_host", "")
        self.to_path = self.config.get("to_path", "~/HyperSwarm/entries")
        self.from_path = self.config.get("from_path") or self.to_path
        self.only_on_host = self.config.get("only_on_host")
        # Allow callers (and tests) to substitute the rsync binary or wrap it.
        self._rsync_cmd = self.config.get("rsync_cmd", "rsync")

    # --------------------------------------------------------------- gates
    def _enabled_here(self) -> bool:
        """Honour the only_on_host gate so a single config can hold sync rules
        for multiple hosts and each host runs only its own.

        Uses host identity (which respects $HYPERSWARM_HOST_IDENTITY override)
        rather than raw socket.gethostname()."""
        if not self.only_on_host:
            return True
        return get_host_identity() == self.only_on_host

    # ------------------------------------------------------------- pushes
    def push(self) -> int:
        if not self._enabled_here():
            return 0
        return self._run(
            local=os.path.expanduser(self.from_path),
            remote=f"{self.to_host}:{self.to_path}",
            direction="push",
        )

    def pull(self) -> int:
        if not self._enabled_here():
            return 0
        return self._run(
            local=os.path.expanduser(self.to_path),
            remote=f"{self.to_host}:{self.from_path}",
            direction="pull",
        )

    # ------------------------------------------------------------- runner
    def _run(self, local: str, remote: str, direction: str) -> int:
        if not self.to_host:
            raise RuntimeError("rsync_ssh: to_host is required")

        # Ensure trailing slash semantics: rsync src/ ⇒ contents-of-src
        local_arg = local.rstrip("/") + "/"
        remote_arg = remote.rstrip("/") + "/"

        if direction == "push":
            src, dst = local_arg, remote_arg
            # Make sure the local source exists before pushing.
            Path(local).mkdir(parents=True, exist_ok=True)
        else:
            src, dst = remote_arg, local_arg
            Path(local).mkdir(parents=True, exist_ok=True)

        cmd = [
            self._rsync_cmd,
            "-az",                  # archive + compress
            "--itemize-changes",    # list each file we touch (for the count)
            "--prune-empty-dirs",
            src,
            dst,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(
                f"rsync {direction} failed (exit {result.returncode}): "
                f"{result.stderr.strip()[:300]}"
            )
        return self._count_transferred(result.stdout)

    @staticmethod
    def _count_transferred(stdout: str) -> int:
        """rsync --itemize-changes prints one line per touched item starting
        with a marker like '<f' (file send), '>f' (file receive), etc. Count
        only files (the 'f' second char), skipping directories ('d')."""
        n = 0
        for line in stdout.splitlines():
            if len(line) >= 2 and line[0] in "<>" and line[1] == "f":
                n += 1
        return n

    # Convenience for debugging — not part of the Sync interface.
    def render_command(self, direction: str) -> str:
        local = os.path.expanduser(self.from_path if direction == "push" else self.to_path)
        if direction == "push":
            return shlex.join([self._rsync_cmd, "-az", local + "/", f"{self.to_host}:{self.to_path}/"])
        return shlex.join([self._rsync_cmd, "-az", f"{self.to_host}:{self.from_path}/", local + "/"])
