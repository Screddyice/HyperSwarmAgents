"""Tests for MLXLoRATrainer.

The actual mlx_lm subprocess can't run in CI (needs Apple Silicon and a real
HF model download). Tests exercise the orchestration only — corpus splitting,
command construction, threshold/concurrency guards, success-path state writing,
failure-mode state preservation — by injecting a `runner` callable that
simulates whatever subprocess outcome we want.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperswarm.tuners.lora_mlx import MLXLoRATrainer, BACKEND_NAME


def _write_corpus(path: Path, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i in range(n):
            f.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "system", "content": "sys"},
                            {"role": "user", "content": f"q{i} long enough"},
                            {"role": "assistant", "content": f"a{i} long enough"},
                        ]
                    }
                )
                + "\n"
            )


def _runner_writes_adapter(*, cmd, cwd, log_path):
    """Simulate a successful mlx_lm.lora run: write the adapter safetensors
    file the real trainer would emit, return exit code 0."""
    # Find --adapter-path in cmd
    idx = cmd.index("--adapter-path")
    adapter_dir = Path(cmd[idx + 1])
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapters.safetensors").write_bytes(b"fake-weights")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("Iter 600/600: train loss 0.42")
    return 0


def _runner_fails(*, cmd, cwd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("CUDA OOM (ironic but possible)")
    return 1


def _runner_writes_no_adapter(*, cmd, cwd, log_path):
    """Simulate a 'success' exit code but the adapter dir is empty — possible
    if mlx_lm changes its output convention. Trainer should treat this as a
    failure rather than blindly recording a missing adapter."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("trained but wrote nothing somehow")
    return 0


# --- Tests ------------------------------------------------------------------


def test_train_skips_empty_corpus(tmp_path: Path):
    t = MLXLoRATrainer(
        agent="jarvis",
        corpus_base=tmp_path / "tune",
        state_dir=tmp_path / "state",
        runner=_runner_writes_adapter,
    )
    res = t.train()
    assert res["status"] == "skipped"
    assert "empty corpus" in res["reason"]


def test_train_skips_below_threshold(tmp_path: Path):
    _write_corpus(tmp_path / "tune" / "jarvis" / "corpus.jsonl", n=10)
    t = MLXLoRATrainer(
        agent="jarvis",
        corpus_base=tmp_path / "tune",
        state_dir=tmp_path / "state",
        min_new_examples=50,
        runner=_runner_writes_adapter,
    )
    res = t.train()
    assert res["status"] == "skipped"
    assert "10 new examples" in res["reason"]


def test_train_completes_writes_adapter(tmp_path: Path):
    _write_corpus(tmp_path / "tune" / "jarvis" / "corpus.jsonl", n=100)
    t = MLXLoRATrainer(
        agent="jarvis",
        corpus_base=tmp_path / "tune",
        state_dir=tmp_path / "state",
        min_new_examples=50,
        runner=_runner_writes_adapter,
    )
    res = t.train()
    assert res["status"] == "completed"
    assert res["backend"] == BACKEND_NAME
    assert "adapter" in res["adapter_path"]
    assert Path(res["adapter_path"], "adapters.safetensors").exists()
    state = json.loads((tmp_path / "state" / "jarvis" / "finetune-state.json").read_text())
    assert state["backend"] == BACKEND_NAME
    assert state["current_adapter"] == res["adapter_path"]
    assert state["last_run_status"] == "completed"
    assert state["history"][0]["status"] == "completed"
    assert state["history"][0]["base_model"] == "Qwen/Qwen3-8B"
    assert state["history"][0]["fine_tune_type"] == "lora"


def test_train_split_train_valid(tmp_path: Path):
    """val_fraction=0.1 → ~10% of examples go to valid.jsonl."""
    _write_corpus(tmp_path / "tune" / "jarvis" / "corpus.jsonl", n=100)
    t = MLXLoRATrainer(
        agent="jarvis",
        corpus_base=tmp_path / "tune",
        state_dir=tmp_path / "state",
        min_new_examples=50,
        runner=_runner_writes_adapter,
    )
    res = t.train()
    assert res["n_train"] + res["n_valid"] == 100
    assert res["n_valid"] >= 1
    # Verify the split files actually exist on disk
    run_dir = Path(res["adapter_path"]).parent
    assert (run_dir / "data" / "train.jsonl").exists()
    assert (run_dir / "data" / "valid.jsonl").exists()


def test_train_command_includes_correct_flags(tmp_path: Path):
    """Verify the constructed command line matches what mlx_lm.lora expects."""
    _write_corpus(tmp_path / "tune" / "jarvis" / "corpus.jsonl", n=100)
    captured: list[list[str]] = []

    def capture_runner(*, cmd, cwd, log_path):
        captured.append(cmd)
        return _runner_writes_adapter(cmd=cmd, cwd=cwd, log_path=log_path)

    t = MLXLoRATrainer(
        agent="jarvis",
        base_model="Qwen/Qwen3-8B",
        num_layers=8,
        iters=200,
        learning_rate=2e-5,
        corpus_base=tmp_path / "tune",
        state_dir=tmp_path / "state",
        min_new_examples=50,
        runner=capture_runner,
    )
    t.train()
    assert len(captured) == 1
    cmd = captured[0]
    assert "--train" in cmd
    assert "--model" in cmd and cmd[cmd.index("--model") + 1] == "Qwen/Qwen3-8B"
    assert cmd[cmd.index("--num-layers") + 1] == "8"
    assert cmd[cmd.index("--iters") + 1] == "200"
    assert cmd[cmd.index("--learning-rate") + 1] == "2e-05"
    assert cmd[cmd.index("--fine-tune-type") + 1] == "lora"


def test_train_failure_marks_failed_and_preserves_prior_adapter(tmp_path: Path):
    """A failed run should NOT overwrite the current_adapter from a prior
    successful run. Symmetric with the Unsloth trainer's behavior."""
    _write_corpus(tmp_path / "tune" / "jarvis" / "corpus.jsonl", n=100)
    t1 = MLXLoRATrainer(
        agent="jarvis",
        corpus_base=tmp_path / "tune",
        state_dir=tmp_path / "state",
        min_new_examples=50,
        runner=_runner_writes_adapter,
    )
    res1 = t1.train()
    first_adapter = res1["adapter_path"]

    _write_corpus(tmp_path / "tune" / "jarvis" / "corpus.jsonl", n=200)
    t2 = MLXLoRATrainer(
        agent="jarvis",
        corpus_base=tmp_path / "tune",
        state_dir=tmp_path / "state",
        min_new_examples=50,
        runner=_runner_fails,
    )
    res2 = t2.train()
    assert res2["status"] == "failed"
    state = json.loads((tmp_path / "state" / "jarvis" / "finetune-state.json").read_text())
    assert state["current_adapter"] == first_adapter
    assert state["last_run_status"] == "failed"
    assert len(state["history"]) == 2
    assert state["history"][1]["status"] == "failed"


def test_train_subprocess_exit_zero_but_no_adapter_marks_failed(tmp_path: Path):
    """Defensive check: if mlx_lm exits 0 but writes no adapter file, we
    treat it as a failure rather than recording a phantom adapter."""
    _write_corpus(tmp_path / "tune" / "jarvis" / "corpus.jsonl", n=100)
    t = MLXLoRATrainer(
        agent="jarvis",
        corpus_base=tmp_path / "tune",
        state_dir=tmp_path / "state",
        min_new_examples=50,
        runner=_runner_writes_no_adapter,
    )
    res = t.train()
    assert res["status"] == "failed"
    assert "no adapter files" in res["error"]


def test_train_skips_when_previous_run_marked_running(tmp_path: Path):
    _write_corpus(tmp_path / "tune" / "jarvis" / "corpus.jsonl", n=100)
    state_path = tmp_path / "state" / "jarvis" / "finetune-state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "backend": BACKEND_NAME,
                "model_base": "Qwen/Qwen3-8B",
                "last_run_id": "2026-05-10-abc",
                "last_run_status": "running",
                "last_examples": 0,
                "current_adapter": None,
                "current_gguf": None,
                "history": [],
            }
        )
    )
    t = MLXLoRATrainer(
        agent="jarvis",
        corpus_base=tmp_path / "tune",
        state_dir=tmp_path / "state",
        min_new_examples=50,
        runner=_runner_writes_adapter,
    )
    res = t.train()
    assert res["status"] == "skipped"
    assert "still" in res["reason"]


def test_status_returns_current_adapter_after_success(tmp_path: Path):
    _write_corpus(tmp_path / "tune" / "jarvis" / "corpus.jsonl", n=100)
    t = MLXLoRATrainer(
        agent="jarvis",
        corpus_base=tmp_path / "tune",
        state_dir=tmp_path / "state",
        min_new_examples=50,
        runner=_runner_writes_adapter,
    )
    t.train()
    s = t.status()
    assert s["last_run_status"] == "completed"
    assert "adapter" in s["current_adapter"]
    assert s["history_length"] == 1
    assert s["backend"] == BACKEND_NAME


def test_status_no_run_yet(tmp_path: Path):
    t = MLXLoRATrainer(
        agent="jarvis",
        corpus_base=tmp_path / "tune",
        state_dir=tmp_path / "state",
    )
    s = t.status()
    assert s["last_run_id"] is None
    assert s["current_adapter"] is None
    assert s["history_length"] == 0
