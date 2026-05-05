"""GitRemoteScope — assign a scope tag from the GitHub org of the repo at cwd.

Walks up from `entry.cwd` to find the enclosing git repo, reads its `origin`
remote URL, and maps the GitHub org to a configured tag. Also sets
`entry.project` to the repo name as a side effect.

Falls back to internal path-prefix and hostname rules when cwd has no git
remote (e.g. ad-hoc work, unprovisioned workspaces).

Config shape:

    [scope]
    type = "git_remote"
    fallback = "cross"

    [[scope.git_remote]]
    github_org = "teamnebula-ai"
    tag = "NEB"

    [[scope.git_remote]]
    github_org = "mycliqk"
    tag = "Cliqk"

    # Optional fallback rules — same semantics as PathPrefixScope.
    [[scope.path_prefix]]
    prefix = "~/projects/teamnebula.ai"
    tag = "NEB"

    [[scope.hostname]]
    name = "neb-server"
    tag = "NEB"

Order of resolution: hostname → git_remote → path_prefix → fallback.
"""
from __future__ import annotations

import os
import re
import subprocess

from hyperswarm.core.entry import Entry
from hyperswarm.core.host import get_host_identity
from hyperswarm.core.scope import Scope


# Matches both forms of GitHub remote URL:
#   git@github.com:org/repo.git
#   https://github.com/org/repo.git
#   ssh://git@github.com/org/repo.git
_REMOTE_RE = re.compile(
    r"""
    (?:git@|https?://|ssh://(?:git@)?)
    github\.com[:/]
    (?P<org>[^/]+)
    /
    (?P<repo>[^/]+?)
    (?:\.git)?
    /?$
    """,
    re.VERBOSE,
)


class GitRemoteScope(Scope):
    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self._org_rules = list(self.config.get("git_remote", []))
        self._path_rules = list(self.config.get("path_prefix", []))
        self._host_rules = list(self.config.get("hostname", []))
        self._fallback = self.config.get("fallback", "")
        # Cache: cwd → (org, repo) | None. Populated lazily.
        self._cwd_cache: dict[str, tuple[str, str] | None] = {}

    def tag(self, entry: Entry) -> str:
        # Hostname wins — server-side captures pin to their host's company.
        host = get_host_identity()
        for rule in self._host_rules:
            if rule.get("name") and rule["name"] == host:
                return rule.get("tag", self._fallback)

        cwd = os.path.expanduser(entry.cwd or "") or ""

        # Try git remote first.
        remote = self._get_origin(cwd)
        if remote is not None:
            org, repo = remote
            entry.project = repo  # side effect — project lives on Entry
            for rule in self._org_rules:
                if rule.get("github_org") == org:
                    return rule.get("tag", self._fallback)
            # Repo exists but org isn't mapped — fall through to path rules.

        # Path-prefix fallback.
        for rule in self._path_rules:
            prefix = os.path.expanduser(rule.get("prefix", ""))
            if prefix and cwd.startswith(prefix):
                return rule.get("tag", self._fallback)

        return self._fallback

    def _get_origin(self, cwd: str) -> tuple[str, str] | None:
        """Return (org, repo) for the git repo enclosing cwd, or None.

        Cached per-cwd. Walks upward to find `.git`, then asks git for the
        origin URL — covers worktrees, submodules, and gitdir-redirected
        layouts that direct .git/config parsing wouldn't.
        """
        if cwd in self._cwd_cache:
            return self._cwd_cache[cwd]

        git_root = self._find_git_root(cwd)
        if git_root is None:
            self._cwd_cache[cwd] = None
            return None

        try:
            out = subprocess.run(
                ["git", "-C", git_root, "remote", "get-url", "origin"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            self._cwd_cache[cwd] = None
            return None
        if out.returncode != 0:
            self._cwd_cache[cwd] = None
            return None

        url = out.stdout.strip()
        m = _REMOTE_RE.match(url)
        if not m:
            self._cwd_cache[cwd] = None
            return None

        result = (m.group("org"), m.group("repo"))
        self._cwd_cache[cwd] = result
        return result

    @staticmethod
    def _find_git_root(cwd: str) -> str | None:
        """Walk up from cwd until we find a `.git` (dir or file). Returns the
        directory containing it, or None if we hit the filesystem root.
        """
        if not cwd:
            return None
        path = os.path.abspath(cwd)
        while True:
            if os.path.exists(os.path.join(path, ".git")):
                return path
            parent = os.path.dirname(path)
            if parent == path:
                return None
            path = parent
