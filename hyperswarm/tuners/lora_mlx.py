"""LoRA fine-tune via Apple's MLX framework — primary backend on macOS.

Why this exists alongside the Unsloth/CUDA backend:

- Apple Silicon (M1-M5+) has unified memory, so an M-series Max with 32-64GB
  RAM can LoRA-tune an 8B model without spending a dollar on cloud GPU
- MLX uses Metal Performance Shaders directly; faster than CPU and
  zero-config compared to setting up CUDA elsewhere
- Free + private + always-available on the user's primary workstation

When to use this vs `lora_local.py` (Unsloth/CUDA):
- Default to MLX on macOS arm64 (this module)
- Fall back to Unsloth on Linux + CUDA (the other module)
- The CLI's `tune-train-local` command auto-detects via platform/arch sniff

Wraps `python -m mlx_lm lora --train` as a subprocess. We use the CLI rather
than the Python API because mlx_lm's API surface still churns release-to-release
and the CLI is explicitly stable. The cost is one extra process spawn per
training cycle, which is trivial compared to the actual training time.

State file shape mirrors `lora_local.py` exactly so any downstream router
that reads `current_adapter` / `current_gguf` / `backend` keeps working
across backend swaps.
"""
from __future__ import annotations

import datetime
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

DEFAULT_STATE_DIR = "~/.local/state/hyperswarm/tune"
DEFAULT_CORPUS_BASE = "~/.openclaw/tune"
DEFAULT_BASE_MODEL = "Qwen/Qwen3-8B"
DEFAULT_LORA_LAYERS = 16
DEFAULT_BATCH_SIZE = 1
DEFAULT_ITERS = 600
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_MAX_SEQ_LENGTH = 4096
DEFAULT_VAL_FRACTION = 0.1
DEFAULT_GRAD_ACCUMULATION_STEPS = 4
DEFAULT_MIN_NEW_EXAMPLES = 50
DEFAULT_SEED = 42
BACKEND_NAME = "lora-mlx"


def _expand(p: str | os.PathLike) -> Path:
    return Path(os.path.expanduser(str(p)))


def _new_run_id() -> str:
    return f"{datetime.date.today().isoformat()}-{uuid.uuid4().hex[:6]}"


def is_mlx_available() -> bool:
    """True iff this host can run mlx_lm training (macOS arm64 with mlx-lm
    installed). Used by backend auto-detection in the CLI."""
    if platform.system() != "Darwin":
        return False
    if platform.machine() not in ("arm64", "aarch64"):
        return False
    try:
        import mlx_lm  # noqa: F401
        return True
    except ImportError:
        return False


@dataclass
class MLXLoRATrainer:
    agent: str
    base_model: str = DEFAULT_BASE_MODEL
    num_layers: int = DEFAULT_LORA_LAYERS
    batch_size: int = DEFAULT_BATCH_SIZE
    iters: int = DEFAULT_ITERS
    learning_rate: float = DEFAULT_LEARNING_RATE
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH
    grad_accumulation_steps: int = DEFAULT_GRAD_ACCUMULATION_STEPS
    val_fraction: float = DEFAULT_VAL_FRACTION
    seed: int = DEFAULT_SEED
    fine_tune_type: str = "lora"  # lora | dora | full
    min_new_examples: int = DEFAULT_MIN_NEW_EXAMPLES
    corpus_base: Path | None = None
    state_dir: Path | None = None
    output_base: Path | None = None
    python_exec: str = sys.executable

    # Test/inject hook. Real subprocess invocation goes through this so we can
    # exercise the orchestration in tests without actually training.
    runner: callable | None = None

    def __post_init__(self) -> None:
        self.corpus_base = _expand(self.corpus_base or DEFAULT_CORPUS_BASE)
        self.state_dir = _expand(self.state_dir or DEFAULT_STATE_DIR)
        self.output_base = _expand(
            self.output_base or (self.corpus_base / self.agent / "lora-output-mlx")
        )

    @property
    def corpus_path(self) -> Path:
        return self.corpus_base / self.agent / "corpus.jsonl"

    @property
    def state_path(self) -> Path:
        return self.state_dir / self.agent / "finetune-state.json"

    def _load_state(self) -> dict:
        if self.state_path.exists():
            try:
                d = json.loads(self.state_path.read_text())
                # Don't blow away an existing backend tag — we want history
                # entries to record which backend trained which run.
                d.setdefault("backend", BACKEND_NAME)
                return d
            except (ValueError, json.JSONDecodeError):
                pass
        return {
            "backend": BACKEND_NAME,
            "model_base": self.base_model,
            "last_run_id": None,
            "last_run_status": None,
            "last_examples": 0,
            "current_adapter": None,
            "current_gguf": None,
            "history": [],
        }

    def _save_state(self, state: dict) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2))

    def _count_examples(self) -> int:
        if not self.corpus_path.exists():
            return 0
        with open(self.corpus_path) as f:
            return sum(1 for line in f if line.strip())

    def status(self) -> dict:
        s = self._load_state()
        return {
            "backend": s.get("backend"),
            "agent": self.agent,
            "last_run_id": s.get("last_run_id"),
            "last_run_status": s.get("last_run_status"),
            "current_adapter": s.get("current_adapter"),
            "current_gguf": s.get("current_gguf"),
            "history_length": len(s.get("history", [])),
        }

    def _split_corpus(self, run_dir: Path) -> tuple[Path, int, int]:
        """Random-split corpus.jsonl into train.jsonl + valid.jsonl in run_dir.

        Returns (data_dir, n_train, n_valid). MLX expects both files; even with
        val_fraction=0 we write a 1-line valid.jsonl so the CLI doesn't error.
        """
        data_dir = run_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        with open(self.corpus_path) as f:
            lines = [ln for ln in f if ln.strip()]
        rng = random.Random(self.seed)
        rng.shuffle(lines)
        n_total = len(lines)
        n_valid = max(1, int(round(n_total * self.val_fraction))) if n_total > 1 else 0
        valid_lines = lines[:n_valid]
        train_lines = lines[n_valid:] or lines  # never leave train empty
        with open(data_dir / "train.jsonl", "w") as f:
            f.writelines(train_lines)
        with open(data_dir / "valid.jsonl", "w") as f:
            f.writelines(valid_lines or train_lines[:1])
        return data_dir, len(train_lines), len(valid_lines)

    def _build_command(self, *, data_dir: Path, adapter_dir: Path) -> list[str]:
        """Construct the mlx_lm.lora subprocess argv."""
        return [
            self.python_exec, "-m", "mlx_lm", "lora",
            "--train",
            "--model", self.base_model,
            "--data", str(data_dir),
            "--fine-tune-type", self.fine_tune_type,
            "--num-layers", str(self.num_layers),
            "--batch-size", str(self.batch_size),
            "--iters", str(self.iters),
            "--learning-rate", str(self.learning_rate),
            "--max-seq-length", str(self.max_seq_length),
            "--grad-accumulation-steps", str(self.grad_accumulation_steps),
            "--adapter-path", str(adapter_dir),
            "--seed", str(self.seed),
        ]

    def _real_run(self, *, cmd: list[str], cwd: Path, log_path: Path) -> int:
        """Run the mlx_lm.lora subprocess; tee output to log file; return exit code."""
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as logf:
            logf.write(f"$ {' '.join(cmd)}\n\n")
            logf.flush()
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                stdout=logf,
                stderr=subprocess.STDOUT,
                check=False,
            )
        return proc.returncode

    def train(self) -> dict:
        """Run one MLX LoRA training pass. Idempotent guards: empty corpus,
        threshold, previous run still claimed-as-running."""
        state = self._load_state()
        last_status = state.get("last_run_status")
        active = {"queued", "running"}
        if last_status in active:
            return {
                "status": "skipped",
                "reason": f"previous run still {last_status}",
                "last_run_id": state.get("last_run_id"),
            }
        total = self._count_examples()
        delta = total - state.get("last_examples", 0)
        if total == 0:
            return {"status": "skipped", "reason": "empty corpus"}
        if delta < self.min_new_examples:
            return {
                "status": "skipped",
                "reason": f"only {delta} new examples (threshold {self.min_new_examples})",
                "total_examples": total,
            }

        run_id = _new_run_id()
        run_dir = self.output_base / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        adapter_dir = run_dir / "adapter"
        log_path = run_dir / "train.log"

        data_dir, n_train, n_valid = self._split_corpus(run_dir)

        state["last_run_id"] = run_id
        state["last_run_status"] = "running"
        state["last_examples"] = total
        history_entry = {
            "run_id": run_id,
            "examples": total,
            "delta": delta,
            "n_train": n_train,
            "n_valid": n_valid,
            "base_model": self.base_model,
            "fine_tune_type": self.fine_tune_type,
            "num_layers": self.num_layers,
            "iters": self.iters,
            "learning_rate": self.learning_rate,
            "started_at": int(time.time()),
            "status": "running",
            "adapter_path": None,
            "log_path": str(log_path),
        }
        state["history"].append(history_entry)
        self._save_state(state)

        cmd = self._build_command(data_dir=data_dir, adapter_dir=adapter_dir)
        runner = self.runner or self._real_run
        try:
            rc = runner(cmd=cmd, cwd=run_dir, log_path=log_path)
        except Exception as e:
            history_entry["status"] = "failed"
            history_entry["error"] = str(e)[:500]
            history_entry["completed_at"] = int(time.time())
            state["last_run_status"] = "failed"
            self._save_state(state)
            return {"status": "failed", "backend": BACKEND_NAME, "run_id": run_id, "error": str(e)}

        if rc != 0:
            history_entry["status"] = "failed"
            history_entry["exit_code"] = rc
            history_entry["completed_at"] = int(time.time())
            state["last_run_status"] = "failed"
            self._save_state(state)
            return {
                "status": "failed",
                "backend": BACKEND_NAME,
                "run_id": run_id,
                "exit_code": rc,
                "log_path": str(log_path),
            }

        # Verify adapter dir actually got something. mlx_lm writes
        # adapters.safetensors (and optionally checkpoint files) into adapter_dir.
        adapter_files = list(adapter_dir.glob("*.safetensors")) if adapter_dir.exists() else []
        if not adapter_files:
            history_entry["status"] = "failed"
            history_entry["error"] = f"no adapter files written under {adapter_dir}"
            history_entry["completed_at"] = int(time.time())
            state["last_run_status"] = "failed"
            self._save_state(state)
            return {
                "status": "failed",
                "backend": BACKEND_NAME,
                "run_id": run_id,
                "error": history_entry["error"],
            }

        history_entry["status"] = "completed"
        history_entry["adapter_path"] = str(adapter_dir)
        history_entry["completed_at"] = int(time.time())
        state["last_run_status"] = "completed"
        state["current_adapter"] = str(adapter_dir)
        self._save_state(state)
        return {
            "status": "completed",
            "backend": BACKEND_NAME,
            "run_id": run_id,
            "adapter_path": str(adapter_dir),
            "examples": total,
            "n_train": n_train,
            "n_valid": n_valid,
            "log_path": str(log_path),
        }


def train_mlx(agent: str, **kwargs) -> dict:
    return MLXLoRATrainer(agent=agent, **kwargs).train()


def status_mlx(agent: str, **kwargs) -> dict:
    return MLXLoRATrainer(agent=agent, **kwargs).status()
