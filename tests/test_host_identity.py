"""Verify host identity resolution order:
  1. $HYPERSWARM_HOST_IDENTITY env var
  2. ~/.config/hyperswarm/host.identity file
  3. socket.gethostname() fallback
"""
from __future__ import annotations

import socket
from pathlib import Path

from hyperswarm.core import host as host_mod


def test_env_var_wins(monkeypatch):
    monkeypatch.setenv("HYPERSWARM_HOST_IDENTITY", "neb-server")
    assert host_mod.get_host_identity() == "neb-server"


def test_env_var_with_whitespace_is_trimmed(monkeypatch):
    monkeypatch.setenv("HYPERSWARM_HOST_IDENTITY", "  neb-server\n")
    assert host_mod.get_host_identity() == "neb-server"


def test_empty_env_var_falls_through(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERSWARM_HOST_IDENTITY", "   ")
    monkeypatch.setattr(host_mod, "DEFAULT_IDENTITY_FILE", str(tmp_path / "host.identity"))
    # File doesn't exist, env is empty → falls back to gethostname
    assert host_mod.get_host_identity() == socket.gethostname()


def test_file_used_when_env_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("HYPERSWARM_HOST_IDENTITY", raising=False)
    f = tmp_path / "host.identity"
    f.write_text("cliqk-server\n")
    monkeypatch.setattr(host_mod, "DEFAULT_IDENTITY_FILE", str(f))
    assert host_mod.get_host_identity() == "cliqk-server"


def test_env_var_beats_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERSWARM_HOST_IDENTITY", "from-env")
    f = tmp_path / "host.identity"
    f.write_text("from-file\n")
    monkeypatch.setattr(host_mod, "DEFAULT_IDENTITY_FILE", str(f))
    assert host_mod.get_host_identity() == "from-env"


def test_default_falls_back_to_gethostname(tmp_path, monkeypatch):
    monkeypatch.delenv("HYPERSWARM_HOST_IDENTITY", raising=False)
    monkeypatch.setattr(host_mod, "DEFAULT_IDENTITY_FILE", str(tmp_path / "no-such-file"))
    assert host_mod.get_host_identity() == socket.gethostname()
