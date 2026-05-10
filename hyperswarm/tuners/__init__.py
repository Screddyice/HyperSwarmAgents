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

__all__ = [
    "OpenClawCorpusCollector",
    "collect_corpus",
    "LocalLoRATrainer",
    "train_local",
    "status_local",
]

# Backend history note:
#   - Original implementation used OpenAI's hosted fine-tune API. Removed when
#     OpenAI announced wind-down.
#   - Replaced with self-hosted LoRA via Unsloth (Karpathy-style: own model,
#     own data, own training). Runs on a CUDA host; corpus.jsonl format stays
#     identical so the collector and watcher don't change.
