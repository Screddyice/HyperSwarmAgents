"""Verify the PathPrefixScope reference implementation:

  - first matching prefix wins
  - hostname rules override path rules when host matches
  - fallback applies when nothing matches
  - leading ~ in prefix is expanded
"""
from __future__ import annotations

import socket

from hyperswarm.core.entry import Entry
from hyperswarm.scopes.path_prefix import PathPrefixScope


def _entry(cwd: str) -> Entry:
    return Entry(runtime="test", cwd=cwd, summary="", body="")


def test_first_matching_prefix_wins():
    cfg = {
        "path_prefix": [
            {"prefix": "/work/teamnebula.ai", "tag": "NEB"},
            {"prefix": "/work/Cliqk", "tag": "Cliqk"},
            {"prefix": "/work", "tag": "cross"},
        ],
        "fallback": "unscoped",
    }
    scope = PathPrefixScope(cfg)
    assert scope.tag(_entry("/work/teamnebula.ai/api")) == "NEB"
    assert scope.tag(_entry("/work/Cliqk/automation")) == "Cliqk"
    assert scope.tag(_entry("/work/Tools")) == "cross"
    assert scope.tag(_entry("/elsewhere")) == "unscoped"


def test_more_specific_prefix_must_be_listed_first():
    """If a generic prefix is listed before a specific one, the generic wins —
    documents that order matters and acts as a regression guard."""
    cfg = {
        "path_prefix": [
            {"prefix": "/work", "tag": "cross"},
            {"prefix": "/work/teamnebula.ai", "tag": "NEB"},
        ],
    }
    scope = PathPrefixScope(cfg)
    assert scope.tag(_entry("/work/teamnebula.ai/api")) == "cross"


def test_hostname_rule_overrides_path():
    cfg = {
        "path_prefix": [{"prefix": "/", "tag": "personal"}],
        "hostname": [{"name": socket.gethostname(), "tag": "Cliqk"}],
    }
    scope = PathPrefixScope(cfg)
    assert scope.tag(_entry("/anywhere")) == "Cliqk"


def test_tilde_in_prefix_is_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = {"path_prefix": [{"prefix": "~/projects/teamnebula.ai", "tag": "NEB"}]}
    scope = PathPrefixScope(cfg)
    assert scope.tag(_entry(str(tmp_path / "projects/teamnebula.ai/api"))) == "NEB"


def test_empty_config_returns_fallback():
    scope = PathPrefixScope({"fallback": "default"})
    assert scope.tag(_entry("/anywhere")) == "default"
