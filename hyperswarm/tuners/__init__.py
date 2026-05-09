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
from hyperswarm.tuners.openai_finetune import (
    OpenAIFineTuneClient,
    trigger_finetune,
    finetune_status,
)

__all__ = [
    "OpenClawCorpusCollector",
    "collect_corpus",
    "OpenAIFineTuneClient",
    "trigger_finetune",
    "finetune_status",
]
