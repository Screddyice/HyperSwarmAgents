"""Tests for LocalLoRATrainer — orchestration without an actual GPU.

The Unsloth+TRL training pipeline is GPU-dependent and can't run in CI.
Tests inject `train_fn` and `gguf_export_fn` callables to exercise:
- threshold + concurrency guards
- run-id generation, output dir layout
- state-file shape (matches what the deprecated OpenAI client wrote, so any
  router downstream keeps working: backend / current_adapter fields)
- failure handling (status flips to "failed", current_adapter NOT overwritten)
- gguf-export branch
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperswarm.tuners.lora_local import LocalLoRATrainer, BACKEND_NAME


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


def _fake_train_ok(*, run_dir: Path, corpus_path: Path) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = run_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_config.json").write_text("{}")
    return {
        "adapter_path": str(adapter_dir),
        "model": "<fake-model-handle>",
        "tokenizer": "<fake-tokenizer-handle>",
    }


def _fake_train_raises(*, run_dir: Path, corpus_path: Path) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    raise RuntimeError("simulated CUDA OOM")


def _fake_export_gguf_ok(*, model, tokenizer, run_dir: Path) -> str:
    p = run_dir / "fake-model-q4_k_m.gguf"
    p.write_bytes(b"GGUF-fake")
    return str(p)


# --- Tests ------------------------------------------------------------------


def test_train_skips_empty_corpus(tmp_path: Path):
    t = LocalLoRATrainer(
        agent="jarvis",
        corpus_base=tmp_path / "tune",
        state_dir=tmp_path / "state",
        train_fn=_fake_train_ok,
    )
    res = t.train()
    assert res["status"] == "skipped"
    assert "empty corpus" in res["reason"]


def test_train_skips_below_threshold(tmp_path: Path):
    _write_corpus(tmp_path / "tune" / "jarvis" / "corpus.jsonl", n=10)
    t = LocalLoRATrainer(
        agent="jarvis",
        corpus_base=tmp_path / "tune",
        state_dir=tmp_path / "state",
        min_new_examples=50,
        train_fn=_fake_train_ok,
    )
    res = t.train()
    assert res["status"] == "skipped"
    assert "10 new examples" in res["reason"]


def test_train_completes_and_records_adapter(tmp_path: Path):
    _write_corpus(tmp_path / "tune" / "jarvis" / "corpus.jsonl", n=60)
    t = LocalLoRATrainer(
        agent="jarvis",
        corpus_base=tmp_path / "tune",
        state_dir=tmp_path / "state",
        min_new_examples=50,
        train_fn=_fake_train_ok,
    )
    res = t.train()
    assert res["status"] == "completed"
    assert res["backend"] == BACKEND_NAME
    assert "adapter" in res["adapter_path"]
    state = json.loads((tmp_path / "state" / "jarvis" / "finetune-state.json").read_text())
    assert state["backend"] == BACKEND_NAME
    assert state["current_adapter"] == res["adapter_path"]
    assert state["last_run_status"] == "completed"
    assert state["history"][0]["status"] == "completed"
    assert state["history"][0]["base_model"] == "Qwen/Qwen3-8B"


def test_train_with_gguf_export(tmp_path: Path):
    _write_corpus(tmp_path / "tune" / "jarvis" / "corpus.jsonl", n=60)
    t = LocalLoRATrainer(
        agent="jarvis",
        corpus_base=tmp_path / "tune",
        state_dir=tmp_path / "state",
        min_new_examples=50,
        export_gguf=True,
        train_fn=_fake_train_ok,
        gguf_export_fn=_fake_export_gguf_ok,
    )
    res = t.train()
    assert res["status"] == "completed"
    assert res["gguf_path"].endswith(".gguf")
    state = json.loads((tmp_path / "state" / "jarvis" / "finetune-state.json").read_text())
    assert state["current_gguf"] == res["gguf_path"]


def test_train_failure_marks_failed_and_preserves_prior_adapter(tmp_path: Path):
    """A failed second run should NOT overwrite the current_adapter from a
    prior successful run."""
    _write_corpus(tmp_path / "tune" / "jarvis" / "corpus.jsonl", n=60)
    t = LocalLoRATrainer(
        agent="jarvis",
        corpus_base=tmp_path / "tune",
        state_dir=tmp_path / "state",
        min_new_examples=50,
        train_fn=_fake_train_ok,
    )
    # First successful training
    res1 = t.train()
    first_adapter = res1["adapter_path"]
    assert first_adapter

    # Grow corpus + queue a second run that fails
    _write_corpus(tmp_path / "tune" / "jarvis" / "corpus.jsonl", n=120)
    t2 = LocalLoRATrainer(
        agent="jarvis",
        corpus_base=tmp_path / "tune",
        state_dir=tmp_path / "state",
        min_new_examples=50,
        train_fn=_fake_train_raises,
    )
    res2 = t2.train()
    assert res2["status"] == "failed"
    assert "OOM" in res2["error"]

    state = json.loads((tmp_path / "state" / "jarvis" / "finetune-state.json").read_text())
    # Current adapter from FIRST successful run is still there
    assert state["current_adapter"] == first_adapter
    # Last-run status reflects the most recent FAILED attempt
    assert state["last_run_status"] == "failed"
    # History has both
    assert len(state["history"]) == 2
    assert state["history"][0]["status"] == "completed"
    assert state["history"][1]["status"] == "failed"


def test_train_skips_when_previous_run_marked_running(tmp_path: Path):
    """If state was left in 'running' (previous trainer crashed mid-run),
    we don't blindly start a new one until something marks it failed."""
    _write_corpus(tmp_path / "tune" / "jarvis" / "corpus.jsonl", n=60)
    t = LocalLoRATrainer(
        agent="jarvis",
        corpus_base=tmp_path / "tune",
        state_dir=tmp_path / "state",
        min_new_examples=50,
        train_fn=_fake_train_ok,
    )
    # Manually seed state to running
    state_path = tmp_path / "state" / "jarvis" / "finetune-state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "backend": BACKEND_NAME,
                "model_base": "Qwen/Qwen3-8B",
                "last_run_id": "2026-05-09-abc",
                "last_run_status": "running",
                "last_examples": 0,
                "current_adapter": None,
                "current_gguf": None,
                "history": [],
            }
        )
    )
    res = t.train()
    assert res["status"] == "skipped"
    assert "still" in res["reason"]


def test_status_returns_current_adapter_after_success(tmp_path: Path):
    _write_corpus(tmp_path / "tune" / "jarvis" / "corpus.jsonl", n=60)
    t = LocalLoRATrainer(
        agent="jarvis",
        corpus_base=tmp_path / "tune",
        state_dir=tmp_path / "state",
        min_new_examples=50,
        train_fn=_fake_train_ok,
    )
    t.train()
    s = t.status()
    assert s["last_run_status"] == "completed"
    assert "adapter" in s["current_adapter"]
    assert s["history_length"] == 1
    assert s["backend"] == BACKEND_NAME


def test_status_no_run_yet(tmp_path: Path):
    t = LocalLoRATrainer(
        agent="jarvis",
        corpus_base=tmp_path / "tune",
        state_dir=tmp_path / "state",
    )
    s = t.status()
    assert s["last_run_id"] is None
    assert s["current_adapter"] is None
    assert s["history_length"] == 0
