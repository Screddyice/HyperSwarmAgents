"""Self-hosted LoRA fine-tune (Karpathy-style: own model, own data, own training).

Pattern reference: Karpathy's framing of "weights vs context" — this module is
the weights side of the loop. Reads the same OpenAI-chat-format
``corpus.jsonl`` the corpus collector emits, runs Unsloth-based LoRA training
on a CUDA GPU, writes the resulting adapter (and optionally a GGUF export
that Ollama can load directly) into ``~/.openclaw/tune/<agent>/lora-output/``.

Why Unsloth: it gets ~2× speedup and ~70% memory reduction over vanilla
HuggingFace TRL training, which is what makes "fine-tune Qwen3 8B on a
consumer 24GB GPU" actually viable. Underneath it's still SFT (supervised
fine-tuning) — same training paradigm as OpenAI's deprecated endpoint, but
on weights you control.

Where to run this:
- A host with a CUDA GPU (>= 16GB VRAM for 8B base, >= 24GB for headroom)
- NOT on neb-server / cliqk-server / trc-server (CPU-only AWS t3 instances)
- Typical paths: a RunPod / Lambda Labs box pulled for a training run (~$0.50/hr,
  ~30-60 min per job), or an owned consumer GPU (4090 / 5090).

Inputs from the rest of the pipeline:
- ``~/.openclaw/tune/<agent>/corpus.jsonl`` (written by tune-collect)
- ``~/.local/state/hyperswarm/tune/<agent>/corpus-cursors.json``

Outputs:
- ``~/.openclaw/tune/<agent>/lora-output/<run_id>/`` — Unsloth adapter dir
- ``~/.openclaw/tune/<agent>/lora-output/<run_id>/<base>-q4_k_m.gguf`` (optional)
- ``~/.local/state/hyperswarm/tune/<agent>/finetune-state.json``

The state file shape mirrors what the deprecated OpenAI client wrote, so any
downstream router that already understands ``current_model`` keeps working.
"""
from __future__ import annotations

import datetime
import json
import os
import shutil
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

DEFAULT_STATE_DIR = "~/.local/state/hyperswarm/tune"
DEFAULT_CORPUS_BASE = "~/.openclaw/tune"
DEFAULT_BASE_MODEL = "Qwen/Qwen3-8B"
DEFAULT_LORA_RANK = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_N_EPOCHS = 3
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_MIN_NEW_EXAMPLES = 50
DEFAULT_MAX_SEQ_LENGTH = 4096
BACKEND_NAME = "lora-local"


def _expand(p: str | os.PathLike) -> Path:
    return Path(os.path.expanduser(str(p)))


def _new_run_id() -> str:
    return f"{datetime.date.today().isoformat()}-{uuid.uuid4().hex[:6]}"


@dataclass
class LocalLoRATrainer:
    agent: str
    base_model: str = DEFAULT_BASE_MODEL
    lora_rank: int = DEFAULT_LORA_RANK
    lora_alpha: int = DEFAULT_LORA_ALPHA
    lora_dropout: float = DEFAULT_LORA_DROPOUT
    n_epochs: int = DEFAULT_N_EPOCHS
    learning_rate: float = DEFAULT_LEARNING_RATE
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH
    min_new_examples: int = DEFAULT_MIN_NEW_EXAMPLES
    corpus_base: Path | None = None
    state_dir: Path | None = None
    output_base: Path | None = None
    export_gguf: bool = False
    gguf_quantization: str = "q4_k_m"

    # Test/inject hooks. Real training pipeline goes through these so we can
    # exercise the orchestration in tests without an actual GPU.
    train_fn: callable | None = None
    gguf_export_fn: callable | None = None

    def __post_init__(self) -> None:
        self.corpus_base = _expand(self.corpus_base or DEFAULT_CORPUS_BASE)
        self.state_dir = _expand(self.state_dir or DEFAULT_STATE_DIR)
        self.output_base = _expand(
            self.output_base or (self.corpus_base / self.agent / "lora-output")
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
            "backend": BACKEND_NAME,
            "agent": self.agent,
            "last_run_id": s.get("last_run_id"),
            "last_run_status": s.get("last_run_status"),
            "current_adapter": s.get("current_adapter"),
            "current_gguf": s.get("current_gguf"),
            "history_length": len(s.get("history", [])),
        }

    def _real_train(self, *, run_dir: Path, corpus_path: Path) -> dict:
        """Run actual Unsloth training. Imports are inside the function so
        importing this module on a CPU box (e.g. for tests) doesn't crash."""
        try:
            import torch  # noqa: F401 — sanity-check CUDA is around
            from datasets import load_dataset
            from trl import SFTTrainer, SFTConfig
            from unsloth import FastLanguageModel
        except ImportError as e:
            raise RuntimeError(
                "Local LoRA training requires `pip install unsloth trl datasets torch` "
                "on a CUDA-enabled host."
            ) from e
        run_dir.mkdir(parents=True, exist_ok=True)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.base_model,
            max_seq_length=self.max_seq_length,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
        ds = load_dataset("json", data_files=str(corpus_path), split="train")

        def _format(example: dict) -> dict:
            text = tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False
            )
            return {"text": text}

        ds = ds.map(_format)
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=ds,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            args=SFTConfig(
                output_dir=str(run_dir),
                num_train_epochs=self.n_epochs,
                learning_rate=self.learning_rate,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                logging_steps=10,
                save_strategy="epoch",
                bf16=True,
            ),
        )
        trainer.train()
        adapter_dir = run_dir / "adapter"
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        return {"adapter_path": str(adapter_dir), "model": model, "tokenizer": tokenizer}

    def _real_export_gguf(self, *, model, tokenizer, run_dir: Path) -> str:
        """Export merged LoRA→GGUF so Ollama can load it. Unsloth has a helper."""
        gguf_path = run_dir / f"{Path(self.base_model).name}-{self.gguf_quantization}.gguf"
        try:
            model.save_pretrained_gguf(
                str(run_dir),
                tokenizer,
                quantization_method=self.gguf_quantization,
            )
        except Exception as e:
            raise RuntimeError(f"GGUF export failed: {e}") from e
        # Unsloth writes the gguf next to adapter; locate the actual filename
        for f in run_dir.iterdir():
            if f.suffix == ".gguf":
                return str(f)
        return str(gguf_path)  # fall through; let caller verify existence

    def train(self) -> dict:
        """Run one training pass. Idempotent guards: empty corpus, threshold,
        previous run still claimed-as-running."""
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
        state["last_run_id"] = run_id
        state["last_run_status"] = "running"
        state["last_examples"] = total
        history_entry = {
            "run_id": run_id,
            "examples": total,
            "delta": delta,
            "base_model": self.base_model,
            "lora_rank": self.lora_rank,
            "n_epochs": self.n_epochs,
            "learning_rate": self.learning_rate,
            "started_at": int(time.time()),
            "status": "running",
            "adapter_path": None,
            "gguf_path": None,
        }
        state["history"].append(history_entry)
        self._save_state(state)

        try:
            train_fn = self.train_fn or self._real_train
            train_result = train_fn(run_dir=run_dir, corpus_path=self.corpus_path)
            adapter_path = train_result["adapter_path"]
            gguf_path: str | None = None
            if self.export_gguf:
                exporter = self.gguf_export_fn or self._real_export_gguf
                gguf_path = exporter(
                    model=train_result.get("model"),
                    tokenizer=train_result.get("tokenizer"),
                    run_dir=run_dir,
                )
            history_entry["status"] = "completed"
            history_entry["adapter_path"] = adapter_path
            history_entry["gguf_path"] = gguf_path
            history_entry["completed_at"] = int(time.time())
            state["last_run_status"] = "completed"
            state["current_adapter"] = adapter_path
            if gguf_path:
                state["current_gguf"] = gguf_path
            self._save_state(state)
            return {
                "status": "completed",
                "backend": BACKEND_NAME,
                "run_id": run_id,
                "adapter_path": adapter_path,
                "gguf_path": gguf_path,
                "examples": total,
            }
        except Exception as e:
            history_entry["status"] = "failed"
            history_entry["error"] = str(e)[:500]
            state["last_run_status"] = "failed"
            self._save_state(state)
            return {
                "status": "failed",
                "backend": BACKEND_NAME,
                "run_id": run_id,
                "error": str(e),
            }


def train_local(agent: str, **kwargs) -> dict:
    return LocalLoRATrainer(agent=agent, **kwargs).train()


def status_local(agent: str, **kwargs) -> dict:
    return LocalLoRATrainer(agent=agent, **kwargs).status()
