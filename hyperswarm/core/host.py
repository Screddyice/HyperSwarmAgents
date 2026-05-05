"""Resolve the current host's identity for scope and sync rules.

Why this exists: a host's `socket.gethostname()` is often something opaque
like `ip-172-31-11-55` (AWS), `runner-abc123` (CI), or `mac-pro-2.local`.
Operators want to think in terms of stable identifiers like `neb-server`,
`cliqk-server`, or `shawn-mac` — names that don't shift when the underlying
machine is reimaged or the DNS suffix changes.

Resolution order (first non-empty wins):
  1. $HYPERSWARM_HOST_IDENTITY environment variable
  2. Contents of ~/.config/hyperswarm/host.identity (single line, trimmed)
  3. socket.gethostname()

The first two are operator-controlled; the last is a sensible default.
"""
from __future__ import annotations

import os
import socket
from pathlib import Path

ENV_VAR = "HYPERSWARM_HOST_IDENTITY"
DEFAULT_IDENTITY_FILE = "~/.config/hyperswarm/host.identity"


def get_host_identity() -> str:
    env = os.environ.get(ENV_VAR, "").strip()
    if env:
        return env

    p = Path(os.path.expanduser(DEFAULT_IDENTITY_FILE))
    if p.exists():
        try:
            value = p.read_text().strip()
            if value:
                return value
        except OSError:
            pass

    return socket.gethostname()
