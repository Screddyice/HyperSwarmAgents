from __future__ import annotations

from hyperswarm.tuners.jarvis_merge import default_sources


def test_default_sources_exclude_personal_home():
    sources = default_sources()
    assert {s.host for s in sources} == {"neb-server", "cliqk-server", "trc-server"}
    assert all(not s.is_local for s in sources)
    assert all("screddy" not in s.remote_path.lower() for s in sources)
    assert all(".claude" not in s.remote_path.lower() for s in sources)
