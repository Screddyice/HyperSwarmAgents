"""Verify GitRemoteScope behavior:

  - hostname rules win over everything else
  - SSH and HTTPS GitHub URLs both parse
  - org maps to configured tag, project is set to repo name
  - cwd inside a subdirectory of the repo still resolves
  - unmapped org falls through to path_prefix
  - no git repo falls through to path_prefix
  - fallback applies when nothing matches
"""
from __future__ import annotations

import subprocess

import pytest

from hyperswarm.core.entry import Entry
from hyperswarm.scopes.git_remote import GitRemoteScope, _REMOTE_RE


def _entry(cwd: str) -> Entry:
    return Entry(runtime="test", cwd=cwd, summary="", body="")


def _init_repo(path, origin_url: str) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(
        ["git", "remote", "add", "origin", origin_url],
        cwd=path, check=True,
    )


def test_remote_re_parses_ssh_url():
    m = _REMOTE_RE.match("git@github.com:teamnebula-ai/teamnebula.ai.git")
    assert m and m.group("org") == "teamnebula-ai"
    assert m.group("repo") == "teamnebula.ai"


def test_remote_re_parses_https_url():
    m = _REMOTE_RE.match("https://github.com/mycliqk/cliqk-marketing-vehicle.git")
    assert m and m.group("org") == "mycliqk"
    assert m.group("repo") == "cliqk-marketing-vehicle"


def test_remote_re_parses_url_without_dot_git():
    m = _REMOTE_RE.match("https://github.com/Screddyice/HyperSwarmAgents")
    assert m and m.group("org") == "Screddyice"
    assert m.group("repo") == "HyperSwarmAgents"


def test_remote_re_rejects_non_github():
    assert _REMOTE_RE.match("https://gitlab.com/foo/bar.git") is None


def test_org_maps_to_tag_and_sets_project(tmp_path, monkeypatch):
    monkeypatch.setattr("hyperswarm.scopes.git_remote.get_host_identity", lambda: "shawn-mac")
    repo = tmp_path / "teamnebula.ai"
    repo.mkdir()
    _init_repo(repo, "git@github.com:teamnebula-ai/teamnebula.ai.git")

    cfg = {
        "git_remote": [{"github_org": "teamnebula-ai", "tag": "NEB"}],
        "fallback": "cross",
    }
    scope = GitRemoteScope(cfg)
    e = _entry(str(repo))
    assert scope.tag(e) == "NEB"
    assert e.project == "teamnebula.ai"


def test_walks_up_from_subdirectory(tmp_path, monkeypatch):
    monkeypatch.setattr("hyperswarm.scopes.git_remote.get_host_identity", lambda: "shawn-mac")
    repo = tmp_path / "hyper_flow"
    sub = repo / "src" / "components"
    sub.mkdir(parents=True)
    _init_repo(repo, "https://github.com/teamnebula-ai/hyper_flow.git")

    cfg = {
        "git_remote": [{"github_org": "teamnebula-ai", "tag": "NEB"}],
        "fallback": "cross",
    }
    scope = GitRemoteScope(cfg)
    e = _entry(str(sub))
    assert scope.tag(e) == "NEB"
    assert e.project == "hyper_flow"


def test_unmapped_org_falls_through_to_path_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr("hyperswarm.scopes.git_remote.get_host_identity", lambda: "shawn-mac")
    repo = tmp_path / "third-party-tool"
    repo.mkdir()
    _init_repo(repo, "git@github.com:somebody-else/third-party-tool.git")

    cfg = {
        "git_remote": [{"github_org": "teamnebula-ai", "tag": "NEB"}],
        "path_prefix": [{"prefix": str(tmp_path), "tag": "cross"}],
        "fallback": "unscoped",
    }
    scope = GitRemoteScope(cfg)
    e = _entry(str(repo))
    assert scope.tag(e) == "cross"
    # Project is still set — repo name is useful even when org isn't mapped.
    assert e.project == "third-party-tool"


def test_no_git_repo_falls_through_to_path_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr("hyperswarm.scopes.git_remote.get_host_identity", lambda: "shawn-mac")
    plain = tmp_path / "TRC"
    plain.mkdir()

    cfg = {
        "git_remote": [{"github_org": "teamnebula-ai", "tag": "NEB"}],
        "path_prefix": [{"prefix": str(tmp_path), "tag": "TRC"}],
        "fallback": "unscoped",
    }
    scope = GitRemoteScope(cfg)
    e = _entry(str(plain))
    assert scope.tag(e) == "TRC"
    assert e.project == ""


def test_hostname_rule_wins_over_git_remote(tmp_path, monkeypatch):
    monkeypatch.setattr("hyperswarm.scopes.git_remote.get_host_identity", lambda: "neb-server")
    repo = tmp_path / "anything"
    repo.mkdir()
    _init_repo(repo, "git@github.com:mycliqk/something.git")

    cfg = {
        "git_remote": [{"github_org": "mycliqk", "tag": "Cliqk"}],
        "hostname": [{"name": "neb-server", "tag": "NEB"}],
    }
    scope = GitRemoteScope(cfg)
    e = _entry(str(repo))
    assert scope.tag(e) == "NEB"


def test_empty_config_returns_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr("hyperswarm.scopes.git_remote.get_host_identity", lambda: "shawn-mac")
    cfg = {"fallback": "default"}
    scope = GitRemoteScope(cfg)
    e = _entry(str(tmp_path))
    assert scope.tag(e) == "default"


def test_repo_with_no_origin_falls_through(tmp_path, monkeypatch):
    monkeypatch.setattr("hyperswarm.scopes.git_remote.get_host_identity", lambda: "shawn-mac")
    repo = tmp_path / "no-origin"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)

    cfg = {
        "git_remote": [{"github_org": "teamnebula-ai", "tag": "NEB"}],
        "path_prefix": [{"prefix": str(tmp_path), "tag": "personal"}],
        "fallback": "cross",
    }
    scope = GitRemoteScope(cfg)
    e = _entry(str(repo))
    assert scope.tag(e) == "personal"
    assert e.project == ""
