"""Lock in the 2026-06-08 retirement: on-device MLX is the ONLY fine-tune path.

These tests guard against regressions that would re-introduce:
- the dead AWS fleet (neb/cliqk/trc) rsync sources in the Jarvis merger
- a cloud-GPU / Unsloth fallback in the CLI backend resolver

See hyperswarm/tuners/__init__.py and hyperswarm/cli.py retirement comments.
"""
from __future__ import annotations

import pytest

from hyperswarm.tuners.jarvis_merge import default_sources


# ── jarvis_merge: Mac-only default sources ─────────────────────────────


def test_default_sources_is_mac_only():
    """default_sources() must yield exactly one local Mac source — no fleet."""
    sources = default_sources()
    assert len(sources) == 1
    only = sources[0]
    assert only.host == "mac"
    assert only.is_local  # ssh_alias is None → read directly, no rsync


def test_default_sources_drops_dead_fleet():
    """No neb/cliqk/trc rsync sources may sneak back into the defaults."""
    hosts = {s.host for s in default_sources()}
    assert "neb-server" not in hosts
    assert "cliqk-server" not in hosts
    assert "trc-server" not in hosts
    # and none of the defaults is a remote (rsync) source
    assert all(s.is_local for s in default_sources())


# ── CLI backend resolver: MLX only, no cloud fallback ──────────────────


def test_resolve_backend_rejects_unsloth():
    """The retired Unsloth/cloud backend must be refused outright."""
    from hyperswarm.cli import _resolve_train_backend

    with pytest.raises(SystemExit) as ei:
        _resolve_train_backend("unsloth")
    assert "MLX" in str(ei.value) or "mlx" in str(ei.value).lower()


def test_resolve_backend_auto_is_mlx_when_available(monkeypatch):
    """'auto' resolves to mlx when MLX is available — never to a cloud backend."""
    import hyperswarm.tuners.lora_mlx as mlx

    monkeypatch.setattr(mlx, "is_mlx_available", lambda: True)
    from hyperswarm.cli import _resolve_train_backend

    assert _resolve_train_backend("auto") == "mlx"
    assert _resolve_train_backend("mlx") == "mlx"


def test_resolve_backend_raises_when_mlx_unavailable_no_cloud_fallback(monkeypatch):
    """When MLX is unavailable we raise — we do NOT fall back to cloud GPU."""
    import hyperswarm.tuners.lora_mlx as mlx

    monkeypatch.setattr(mlx, "is_mlx_available", lambda: False)
    from hyperswarm.cli import _resolve_train_backend

    with pytest.raises(SystemExit) as ei:
        _resolve_train_backend("auto")
    msg = str(ei.value).lower()
    assert "on-device" in msg or "mlx" in msg


# ── registry: lora_local (Unsloth/CUDA) is unwired ─────────────────────


def test_tuner_registry_does_not_export_local_trainer():
    """The Unsloth trainer must not be re-exported from the tuner registry."""
    import hyperswarm.tuners as t

    assert "LocalLoRATrainer" not in t.__all__
    assert "train_local" not in t.__all__
    assert "status_local" not in t.__all__
    assert "MLXLoRATrainer" in t.__all__  # the surviving on-device trainer
