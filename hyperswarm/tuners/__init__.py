"""Tuners — slow learning. Update model weights, not just context.

A Tuner accumulates user/assistant interaction pairs into a fine-tune corpus,
then triggers an actual training job (OpenAI fine-tune API today; LoRA on
self-hosted later) when enough new examples have accumulated. The resulting
fine-tuned model id is written to a state file so openclaw can route
personalization-flavored queries to it.

Pattern reference: Karpathy's "weights vs context" — Sources and Reflectors
update context (fast learning); Tuners update weights (slow learning).
"""
from hyperswarm.tuners.openclaw_corpus import (
    OpenClawCorpusCollector,
    collect_corpus,
)
from hyperswarm.tuners.lora_local import (
    LocalLoRATrainer,
    train_local,
    status_local,
)
from hyperswarm.tuners.lora_mlx import (
    MLXLoRATrainer,
    train_mlx,
    status_mlx,
    is_mlx_available,
)

__all__ = [
    "OpenClawCorpusCollector",
    "collect_corpus",
    "LocalLoRATrainer",
    "train_local",
    "status_local",
    "MLXLoRATrainer",
    "train_mlx",
    "status_mlx",
    "is_mlx_available",
]

# Backend history + selection:
#   1. OpenAI hosted fine-tune (removed — vendor sunset).
#   2. Unsloth on Linux+CUDA (`lora_local.py`) — secondary, kicks in on GPU
#      servers when the primary Mac is offline / not reachable.
#   3. MLX on macOS arm64 (`lora_mlx.py`) — primary. Free, private, fast on
#      Apple Silicon Max-tier hardware. Auto-selected by the CLI on macOS.
# corpus.jsonl format is identical across backends; the only thing that
# changes per-backend is which trainer module the CLI dispatches to and the
# resulting adapter format on disk (MLX safetensors vs Unsloth-saved adapter).
