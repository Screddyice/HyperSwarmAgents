"""PathPrefixScope — assign a scope tag based on cwd path prefix or hostname.

Config shape:

    [[path_prefix]]
    prefix = "~/projects/teamnebula.ai"
    tag = "NEB"

    [[path_prefix]]
    prefix = "~/projects"
    tag = "personal"

    [[hostname]]
    name = "neb-server"
    tag = "NEB"

Order matters — first match wins, so put more-specific prefixes first.
"""
from __future__ import annotations

import os

from hyperswarm.core.entry import Entry
from hyperswarm.core.host import get_host_identity
from hyperswarm.core.scope import Scope


class PathPrefixScope(Scope):
    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self._path_rules = list(self.config.get("path_prefix", []))
        self._host_rules = list(self.config.get("hostname", []))
        self._fallback = self.config.get("fallback", "")

    def tag(self, entry: Entry) -> str:
        host = get_host_identity()
        for rule in self._host_rules:
            name = rule.get("name", "")
            if name and name == host:
                return rule.get("tag", self._fallback)

        cwd = os.path.expanduser(entry.cwd or "") or ""
        for rule in self._path_rules:
            prefix = os.path.expanduser(rule.get("prefix", ""))
            if prefix and cwd.startswith(prefix):
                return rule.get("tag", self._fallback)

        return self._fallback
