"""RsyncSshSync contract tests with a fake `rsync` binary:
  - push and pull build the right argv with trailing-slash semantics
  - count_transferred parses --itemize-changes correctly
  - only_on_host gate skips sync on the wrong host
  - missing to_host raises clearly
  - non-zero rsync exit raises with a descriptive error
"""
from __future__ import annotations

import os
import socket
import stat
from pathlib import Path

import pytest

from hyperswarm.syncs.rsync_ssh import RsyncSshSync


def _make_fake_rsync(tmp_path: Path, *, exit_code: int = 0, itemized_lines: int = 0) -> str:
    """Return a path to a fake rsync that records argv and emits N itemize lines."""
    fake = tmp_path / "fake-rsync.sh"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        f"echo $@ > {tmp_path}/last-argv.txt\n"
        + "\n".join(f"echo '<f+++++++++ file{i}.md'" for i in range(itemized_lines))
        + "\n"
        + f"exit {exit_code}\n"
    )
    fake.chmod(0o755)
    return str(fake)


def test_push_argv_contains_local_then_remote(tmp_path: Path):
    fake = _make_fake_rsync(tmp_path, itemized_lines=0)
    local = tmp_path / "entries"
    local.mkdir()

    sync = RsyncSshSync({
        "direction": "push",
        "to_host": "shawn-mac",
        "to_path": "~/HyperSwarm/entries",
        "from_path": str(local),
        "rsync_cmd": fake,
    })
    moved = sync.push()
    assert moved == 0

    argv = (tmp_path / "last-argv.txt").read_text().strip()
    assert "shawn-mac:" in argv
    assert str(local) + "/" in argv  # trailing slash on src


def test_pull_argv_inverts_direction(tmp_path: Path):
    fake = _make_fake_rsync(tmp_path, itemized_lines=0)
    local = tmp_path / "entries"

    sync = RsyncSshSync({
        "direction": "pull",
        "to_host": "shawn-mac",
        "to_path": str(local),
        "from_path": "~/HyperSwarm/entries",
        "rsync_cmd": fake,
    })
    sync.pull()
    argv = (tmp_path / "last-argv.txt").read_text().strip()
    # Remote is now src, local is dst
    parts = argv.split()
    src_idx = next(i for i, p in enumerate(parts) if "shawn-mac:" in p)
    dst_idx = next(i for i, p in enumerate(parts) if str(local) in p)
    assert src_idx < dst_idx


def test_count_transferred_parses_itemize_output(tmp_path: Path):
    fake = _make_fake_rsync(tmp_path, itemized_lines=3)
    sync = RsyncSshSync({
        "direction": "push",
        "to_host": "shawn-mac",
        "to_path": str(tmp_path / "remote"),
        "from_path": str(tmp_path / "local"),
        "rsync_cmd": fake,
    })
    moved = sync.push()
    assert moved == 3


def test_only_on_host_gate_blocks_wrong_host(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("hyperswarm.syncs.rsync_ssh.get_host_identity", lambda: "test-host")
    fake = _make_fake_rsync(tmp_path)
    sync = RsyncSshSync({
        "direction": "push",
        "to_host": "shawn-mac",
        "to_path": str(tmp_path / "x"),
        "from_path": str(tmp_path / "y"),
        "rsync_cmd": fake,
        "only_on_host": "different-host",
    })
    assert sync.push() == 0
    # rsync was never invoked
    assert not (tmp_path / "last-argv.txt").exists()


def test_only_on_host_gate_passes_matching_host(tmp_path: Path, monkeypatch):
    # Pin host identity so it matches the only_on_host value regardless of
    # whatever ~/.config/hyperswarm/host.identity says on the dev machine.
    monkeypatch.setattr("hyperswarm.syncs.rsync_ssh.get_host_identity", lambda: "test-host")
    fake = _make_fake_rsync(tmp_path, itemized_lines=1)
    sync = RsyncSshSync({
        "direction": "push",
        "to_host": "shawn-mac",
        "to_path": str(tmp_path / "x"),
        "from_path": str(tmp_path / "y"),
        "rsync_cmd": fake,
        "only_on_host": "test-host",
    })
    assert sync.push() == 1


def test_missing_to_host_raises(tmp_path: Path):
    sync = RsyncSshSync({
        "direction": "push",
        "to_host": "",
        "to_path": str(tmp_path / "x"),
        "from_path": str(tmp_path / "y"),
        "rsync_cmd": "/usr/bin/true",
    })
    with pytest.raises(RuntimeError, match="to_host"):
        sync.push()


def test_rsync_failure_surfaces(tmp_path: Path):
    fake = _make_fake_rsync(tmp_path, exit_code=23)  # rsync's "partial transfer" code
    sync = RsyncSshSync({
        "direction": "push",
        "to_host": "shawn-mac",
        "to_path": str(tmp_path / "x"),
        "from_path": str(tmp_path / "y"),
        "rsync_cmd": fake,
    })
    with pytest.raises(RuntimeError, match="exit 23"):
        sync.push()


def test_invalid_direction_rejected_early():
    with pytest.raises(ValueError, match="direction"):
        RsyncSshSync({"direction": "sideways", "to_host": "x"})
