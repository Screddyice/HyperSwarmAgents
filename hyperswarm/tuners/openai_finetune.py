"""OpenAI fine-tune trigger + state.

Manages the fine-tune lifecycle for one agent's corpus:
  trigger() — upload corpus → create fine-tune job → record job id
  status()  — check job state, capture model id when succeeded

State lives at ~/.local/state/hyperswarm/tune/<agent>/finetune-state.json:

  {
    "model_base":      "gpt-4o-mini-2024-07-18",
    "last_job_id":     "ftjob-abc...",
    "last_job_status": "succeeded",
    "current_model":   "ft:gpt-4o-mini-2024-07-18:org::abc...",
    "last_examples":   312,
    "history": [{job_id, examples, model, created_at, status}, ...]
  }

The trigger logic refuses to start a job when:
- No corpus yet
- New-examples-since-last-tune < threshold (default 50)
- A previous job is still running (status: validating_files | queued | running)

This keeps cost predictable and avoids stacking jobs on top of each other.

The actual personalization routing — i.e. how openclaw decides "use the
fine-tuned model vs the stock Codex model" — is handled by the openclaw
config layer, not here. This module's job is to make the fine-tuned model
*available*; the routing decision lives elsewhere.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

DEFAULT_STATE_DIR = "~/.local/state/hyperswarm/tune"
DEFAULT_CORPUS_BASE = "~/.openclaw/tune"
DEFAULT_MIN_NEW_EXAMPLES = 50
DEFAULT_BASE_MODEL = "gpt-4o-mini-2024-07-18"


def _expand(p: str | os.PathLike) -> Path:
    return Path(os.path.expanduser(str(p)))


@dataclass
class OpenAIFineTuneClient:
    agent: str
    base_model: str = DEFAULT_BASE_MODEL
    corpus_base: Path | None = None
    state_dir: Path | None = None
    min_new_examples: int = DEFAULT_MIN_NEW_EXAMPLES
    openai_client: object | None = None  # injectable for tests

    def __post_init__(self) -> None:
        self.corpus_base = _expand(self.corpus_base or DEFAULT_CORPUS_BASE)
        self.state_dir = _expand(self.state_dir or DEFAULT_STATE_DIR)

    @property
    def corpus_path(self) -> Path:
        return self.corpus_base / self.agent / "corpus.jsonl"

    @property
    def state_path(self) -> Path:
        return self.state_dir / self.agent / "finetune-state.json"

    def _client(self):
        if self.openai_client is not None:
            return self.openai_client
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("openai package not installed") from e
        return OpenAI()

    def _load_state(self) -> dict:
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text())
            except (ValueError, json.JSONDecodeError):
                pass
        return {
            "model_base": self.base_model,
            "last_job_id": None,
            "last_job_status": None,
            "current_model": None,
            "last_examples": 0,
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

    def trigger(self) -> dict:
        """Upload corpus → create fine-tune job. Records the job id but does
        not block on completion. Call status() later to capture the resulting
        model id."""
        state = self._load_state()
        # Guard: previous job still running?
        last_status = state.get("last_job_status")
        if last_status in ("validating_files", "queued", "running"):
            return {
                "status": "skipped",
                "reason": f"previous job still {last_status}",
                "last_job_id": state.get("last_job_id"),
            }
        # Guard: enough new examples?
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
        client = self._client()
        # Upload
        with open(self.corpus_path, "rb") as f:
            file_obj = client.files.create(file=f, purpose="fine-tune")
        # Create job
        job = client.fine_tuning.jobs.create(
            training_file=file_obj.id, model=self.base_model
        )
        state["last_job_id"] = job.id
        state["last_job_status"] = job.status
        state["last_examples"] = total
        state["history"].append(
            {
                "job_id": job.id,
                "training_file": file_obj.id,
                "examples": total,
                "delta": delta,
                "base_model": self.base_model,
                "status": job.status,
                "created_at": int(time.time()),
                "model": None,
            }
        )
        self._save_state(state)
        return {
            "status": "started",
            "job_id": job.id,
            "training_file": file_obj.id,
            "examples": total,
            "delta": delta,
            "base_model": self.base_model,
        }

    def status(self) -> dict:
        """Refresh the last job's status from OpenAI. If the job has succeeded,
        capture the new model id and update state."""
        state = self._load_state()
        job_id = state.get("last_job_id")
        if not job_id:
            return {"status": "no-job", "current_model": state.get("current_model")}
        client = self._client()
        job = client.fine_tuning.jobs.retrieve(job_id)
        state["last_job_status"] = job.status
        # Update history entry too
        for h in reversed(state["history"]):
            if h.get("job_id") == job_id:
                h["status"] = job.status
                if getattr(job, "fine_tuned_model", None):
                    h["model"] = job.fine_tuned_model
                break
        if job.status == "succeeded" and getattr(job, "fine_tuned_model", None):
            state["current_model"] = job.fine_tuned_model
        self._save_state(state)
        return {
            "status": job.status,
            "job_id": job_id,
            "current_model": state.get("current_model"),
            "training_file": getattr(job, "training_file", None),
            "fine_tuned_model": getattr(job, "fine_tuned_model", None),
            "error": getattr(job, "error", None),
        }


def trigger_finetune(agent: str, **kwargs) -> dict:
    return OpenAIFineTuneClient(agent=agent, **kwargs).trigger()


def finetune_status(agent: str, **kwargs) -> dict:
    return OpenAIFineTuneClient(agent=agent, **kwargs).status()
