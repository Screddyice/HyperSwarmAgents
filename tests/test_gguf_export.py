"""Tests for GGUFExporter — two-step pipeline (mlx_lm fuse + llama.cpp convert).

Real subprocess work happens via mlx_lm and llama.cpp's convert script;
neither runs in CI. Tests inject a `runner` callable so we can simulate any
subprocess outcome and verify the orchestration: detection of llama.cpp,
correct cmd construction for both steps, fail-soft when llama.cpp absent
(step-1 work is preserved, clear instructions returned), state-file writes
only on full success.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from hyperswarm.tuners.gguf_export import GGUFExporter


def _seed_state(state_dir: Path, agent: str, *, current_adapter: str | None, base_model: str = "Qwen/Qwen3-8B"):
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / agent).mkdir(parents=True, exist_ok=True)
    (state_dir / agent / "finetune-state.json").write_text(
        json.dumps({
            "backend": "lora-mlx",
            "model_base": base_model,
            "current_adapter": current_adapter,
            "current_gguf": None,
            "history": [],
        })
    )


def _seed_adapter(tmp_path: Path) -> Path:
    adapter_dir = tmp_path / "tune" / "jarvis" / "lora-output-mlx" / "run-1" / "adapter"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapters.safetensors").write_bytes(b"adapter-weights")
    return adapter_dir


def _seed_llama_cpp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Drop a fake convert_hf_to_gguf.py and point HYPERSWARM_LLAMA_CPP_DIR at it."""
    llama_dir = tmp_path / "fake_llama_cpp"
    llama_dir.mkdir()
    converter = llama_dir / "convert_hf_to_gguf.py"
    converter.write_text("# fake llama.cpp converter\n")
    monkeypatch.setenv("HYPERSWARM_LLAMA_CPP_DIR", str(llama_dir))
    return converter


# --- Sample runners ---------------------------------------------------------


def make_two_step_runner(*, fuse_writes_safetensors: bool = True, convert_writes_gguf: bool = True):
    """Returns a runner that simulates: step 1 = fuse, step 2 = convert.
    Detects which step it's running by the cmd contents."""

    def runner(*, cmd, cwd, log_path, append=False):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a" if append else "w") as f:
            f.write("\n$ " + " ".join(cmd) + "\n")
        if "fuse" in cmd:
            # Fuse: write safetensors into --save-path
            i = cmd.index("--save-path")
            fused_dir = Path(cmd[i + 1])
            fused_dir.mkdir(parents=True, exist_ok=True)
            if fuse_writes_safetensors:
                (fused_dir / "model.safetensors").write_bytes(b"fused-weights")
                (fused_dir / "config.json").write_text("{}")
            return 0
        if "convert_hf_to_gguf.py" in " ".join(cmd):
            i = cmd.index("--outfile")
            gguf_path = Path(cmd[i + 1])
            if convert_writes_gguf:
                gguf_path.write_bytes(b"GGUF-fake")
            return 0
        raise AssertionError(f"unexpected cmd: {cmd}")

    return runner


def runner_fuse_fails(*, cmd, cwd, log_path, append=False):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("fuse blew up")
    return 1


def runner_convert_fails(*, cmd, cwd, log_path, append=False):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a" if append else "w") as f:
        f.write("$ " + " ".join(cmd) + "\n")
    if "fuse" in cmd:
        i = cmd.index("--save-path")
        fused_dir = Path(cmd[i + 1])
        fused_dir.mkdir(parents=True, exist_ok=True)
        (fused_dir / "model.safetensors").write_bytes(b"fused")
        return 0
    return 1  # convert fails


# --- Guard tests ------------------------------------------------------------


def test_export_fails_when_no_state(tmp_path: Path):
    res = GGUFExporter(
        agent="jarvis",
        state_dir=tmp_path / "state",
        runner=make_two_step_runner(),
    ).export()
    assert res["status"] == "failed"
    assert "no finetune-state" in res["reason"]


def test_export_fails_when_no_current_adapter(tmp_path: Path):
    _seed_state(tmp_path / "state", "jarvis", current_adapter=None)
    res = GGUFExporter(
        agent="jarvis",
        state_dir=tmp_path / "state",
        runner=make_two_step_runner(),
    ).export()
    assert res["status"] == "failed"
    assert "no current_adapter" in res["reason"]


def test_export_fails_when_adapter_dir_missing(tmp_path: Path):
    _seed_state(tmp_path / "state", "jarvis", current_adapter=str(tmp_path / "ghost"))
    res = GGUFExporter(
        agent="jarvis",
        state_dir=tmp_path / "state",
        runner=make_two_step_runner(),
    ).export()
    assert res["status"] == "failed"
    assert "does not exist" in res["reason"]


# --- Step-1 (fuse) failure modes -------------------------------------------


def test_export_fails_when_fuse_subprocess_returns_nonzero(tmp_path: Path):
    adapter = _seed_adapter(tmp_path)
    _seed_state(tmp_path / "state", "jarvis", current_adapter=str(adapter))
    res = GGUFExporter(
        agent="jarvis",
        state_dir=tmp_path / "state",
        runner=runner_fuse_fails,
    ).export()
    assert res["status"] == "failed"
    assert "mlx_lm fuse returned non-zero" in res["reason"]


def test_export_fails_when_fuse_succeeds_but_writes_nothing(tmp_path: Path):
    adapter = _seed_adapter(tmp_path)
    _seed_state(tmp_path / "state", "jarvis", current_adapter=str(adapter))
    res = GGUFExporter(
        agent="jarvis",
        state_dir=tmp_path / "state",
        runner=make_two_step_runner(fuse_writes_safetensors=False),
    ).export()
    assert res["status"] == "failed"
    assert "no safetensors written" in res["reason"]


# --- Step-2 (llama.cpp convert) ---------------------------------------------


def test_export_fails_with_clear_instructions_when_llama_cpp_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Step 1 succeeds; step 2 should hard-fail-soft with install instructions
    so future-Shawn knows exactly what to install."""
    monkeypatch.delenv("HYPERSWARM_LLAMA_CPP_DIR", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "no_llama_here")
    adapter = _seed_adapter(tmp_path)
    _seed_state(tmp_path / "state", "jarvis", current_adapter=str(adapter))
    res = GGUFExporter(
        agent="jarvis",
        state_dir=tmp_path / "state",
        runner=make_two_step_runner(),
    ).export()
    assert res["status"] == "failed"
    assert "llama.cpp converter not found" in res["reason"]
    assert "git clone" in res["reason"]
    assert res["step_1_completed"] is True


def test_export_uses_llama_cpp_from_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _seed_llama_cpp(tmp_path, monkeypatch)
    adapter = _seed_adapter(tmp_path)
    _seed_state(tmp_path / "state", "jarvis", current_adapter=str(adapter))
    res = GGUFExporter(
        agent="jarvis",
        state_dir=tmp_path / "state",
        runner=make_two_step_runner(),
    ).export()
    assert res["status"] == "completed"
    assert "fake_llama_cpp" in res["converter"]


def test_export_succeeds_full_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _seed_llama_cpp(tmp_path, monkeypatch)
    adapter = _seed_adapter(tmp_path)
    _seed_state(tmp_path / "state", "jarvis", current_adapter=str(adapter), base_model="Qwen/Qwen3-8B")
    res = GGUFExporter(
        agent="jarvis",
        state_dir=tmp_path / "state",
        runner=make_two_step_runner(),
    ).export()
    assert res["status"] == "completed"
    gguf = Path(res["gguf_path"])
    assert gguf.exists()
    assert gguf.read_bytes() == b"GGUF-fake"
    state = json.loads((tmp_path / "state" / "jarvis" / "finetune-state.json").read_text())
    assert state["current_gguf"] == str(gguf)
    assert len(state["gguf_history"]) == 1
    assert state["gguf_history"][0]["base_model"] == "Qwen/Qwen3-8B"


def test_export_failure_in_convert_does_not_overwrite_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _seed_llama_cpp(tmp_path, monkeypatch)
    adapter = _seed_adapter(tmp_path)
    _seed_state(tmp_path / "state", "jarvis", current_adapter=str(adapter))
    res = GGUFExporter(
        agent="jarvis",
        state_dir=tmp_path / "state",
        runner=runner_convert_fails,
    ).export()
    assert res["status"] == "failed"
    assert "llama.cpp convert returned non-zero" in res["reason"]
    state = json.loads((tmp_path / "state" / "jarvis" / "finetune-state.json").read_text())
    assert state.get("current_gguf") is None


def test_export_skips_fuse_when_already_done(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Re-running export shouldn't re-fuse if the fused dir is already populated.
    Saves time on retry-after-llama.cpp-install."""
    _seed_llama_cpp(tmp_path, monkeypatch)
    adapter = _seed_adapter(tmp_path)
    _seed_state(tmp_path / "state", "jarvis", current_adapter=str(adapter))

    # Pre-create the fused dir as if a prior fuse already succeeded
    fused_dir = adapter.parent / "fused"
    fused_dir.mkdir(parents=True)
    (fused_dir / "model.safetensors").write_bytes(b"already-fused")

    fuse_called = [False]

    def runner(*, cmd, cwd, log_path, append=False):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a" if append else "w") as f:
            f.write("ran\n")
        if "fuse" in cmd:
            fuse_called[0] = True
            return 0
        if "convert_hf_to_gguf" in " ".join(cmd):
            i = cmd.index("--outfile")
            Path(cmd[i + 1]).write_bytes(b"GGUF")
            return 0
        return 1

    res = GGUFExporter(
        agent="jarvis",
        state_dir=tmp_path / "state",
        runner=runner,
    ).export()
    assert res["status"] == "completed"
    assert fuse_called[0] is False  # fuse was skipped (dir already populated)
