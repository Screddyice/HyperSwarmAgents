from hyperswarm.syncs.rsync_ssh import RsyncSshSync

SYNC_REGISTRY: dict[str, type] = {
    "rsync_ssh": RsyncSshSync,
}

__all__ = ["RsyncSshSync", "SYNC_REGISTRY"]
