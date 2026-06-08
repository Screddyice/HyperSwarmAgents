"""Tuners — slow learning. Update model weights, not just context.

A Tuner accumulates user/assistant interaction pairs into a fine-tune corpus,
then triggers an actual training job (OpenAI fine-tune API today; LoRA on
self-hosted later) when enough new examples have accumulated. The resulting
fine-tuned model id is written to a state file so openclaw can route
personalization-flavored queries to it.

Pattern reference: Karpathy's "weights vs context" — Sources and Reflectors
update context (fast learning); Tuners update weights (slow learning).
"""
# retired 2026-06-08: openclaw decommissioned, replaced by Hermes memory provider
# retired 2026-06-08: cloud GPU fine-tuning removed — on-device MLX only per Shawn.
#   lora_local.py (Unsloth/CUDA) is kept on disk for git history but is no
#   longer wired into the tuner registry. MLX on Apple Silicon is the only
#   fine-tune path.
from hyperswarm.tuners.lora_mlx import (
    MLXLoRATrainer,
    train_mlx,
    status_mlx,
    is_mlx_available,
)
from hyperswarm.tuners.gguf_export import (
    GGUFExporter,
    export_gguf,
)
from hyperswarm.tuners.jarvis_merge import (
    CorpusSource,
    JarvisCorpusMerger,
    default_sources as default_jarvis_sources,
    merge_jarvis_corpus,
)

__all__ = [
    "MLXLoRATrainer",
    "train_mlx",
    "status_mlx",
    "is_mlx_available",
    "GGUFExporter",
    "export_gguf",
    "CorpusSource",
    "JarvisCorpusMerger",
    "default_jarvis_sources",
    "merge_jarvis_corpus",
]

# Backend history + selection:
#   1. OpenAI hosted fine-tune (removed — vendor sunset).
#   2. Unsloth on Linux+CUDA (`lora_local.py`) — retired 2026-06-08: cloud GPU
#      fine-tuning removed — on-device MLX only per Shawn. File kept for git
#      history but unwired (no longer imported here, no CLI dispatch).
#   3. MLX on macOS arm64 (`lora_mlx.py`) — the ONLY fine-tune path. Free,
#      private, fast on Apple Silicon Max-tier hardware.
# corpus.jsonl format is unchanged; the CLI dispatches to the MLX trainer
# unconditionally and the adapter format on disk is MLX safetensors.
