"""cmd_pull/cmd_push must honour each [[sync]] block's declared direction.

Regression: previously `hyperswarm pull` ran .pull() on every block, so push
blocks (remote `to_path`, local `from_path`) were inverted — mkdir'ing their
remote path locally and reading their local path as if it were remote, which
created stray local dirs and failed with rsync exit 23 ("No such file").
"""
from __future__ import annotations

from hyperswarm import cli


def _cfg() -> dict:
    return {
        "sync": [
            {  # push leg: local -> remote
                "type": "rsync_ssh", "direction": "push", "to_host": "h",
                "from_path": "~/projects/HyperSwarm/entries",
                "to_path": "~/HyperSwarm/entries",
            },
            {  # pull leg: remote -> local
                "type": "rsync_ssh", "direction": "pull", "to_host": "h",
                "from_path": "~/HyperSwarm/entries",
                "to_path": "~/projects/HyperSwarm/entries",
            },
            {  # no direction declared -> runs for both (legacy behaviour)
                "type": "rsync_ssh", "to_host": "h", "to_path": "~/both/entries",
            },
        ]
    }


def test_pull_runs_only_pull_and_directionless_blocks():
    paths = {s.to_path for _, s in cli._build_syncs(_cfg(), "pull")}
    assert paths == {"~/projects/HyperSwarm/entries", "~/both/entries"}


def test_push_runs_only_push_and_directionless_blocks():
    paths = {s.to_path for _, s in cli._build_syncs(_cfg(), "push")}
    assert paths == {"~/HyperSwarm/entries", "~/both/entries"}


def test_no_direction_filter_returns_all_blocks():
    assert len(cli._build_syncs(_cfg())) == 3
