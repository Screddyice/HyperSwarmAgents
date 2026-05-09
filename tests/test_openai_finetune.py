"""Tests for OpenAIFineTuneClient — guards, state persistence, status capture."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from hyperswarm.tuners.openai_finetune import OpenAIFineTuneClient


# --- Mock OpenAI client ------------------------------------------------------


@dataclass
class _MockFile:
    id: str = "file-abc123"


@dataclass
class _MockJob:
    id: str = "ftjob-xyz789"
    status: str = "validating_files"
    fine_tuned_model: str | None = None
    training_file: str = "file-abc123"
    error: Any = None


class _MockFiles:
    def __init__(self, parent):
        self.parent = parent

    def create(self, file, purpose):
        self.parent.uploads_called += 1
        return _MockFile(id=f"file-mock-{self.parent.uploads_called}")


class _MockFineTuningJobs:
    def __init__(self, parent):
        self.parent = parent

    def create(self, training_file, model):
        self.parent.creates_called += 1
        return _MockJob(
            id=f"ftjob-mock-{self.parent.creates_called}",
            status="validating_files",
            training_file=training_file,
        )

    def retrieve(self, job_id):
        # Caller can set parent.next_status / parent.next_model to control
        return _MockJob(
            id=job_id,
            status=self.parent.next_status,
            fine_tuned_model=self.parent.next_model,
            training_file="file-abc123",
            error=None,
        )


class _MockFineTuning:
    def __init__(self, parent):
        self.jobs = _MockFineTuningJobs(parent)


class MockOpenAI:
    def __init__(self):
        self.uploads_called = 0
        self.creates_called = 0
        self.next_status = "running"
        self.next_model = None
        self.files = _MockFiles(self)
        self.fine_tuning = _MockFineTuning(self)


def _write_corpus(path: Path, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i in range(n):
            f.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "system", "content": "sys"},
                            {"role": "user", "content": f"q{i} is long enough"},
                            {"role": "assistant", "content": f"a{i} is also long enough"},
                        ]
                    }
                )
                + "\n"
            )


# --- Tests ------------------------------------------------------------------


def test_trigger_skips_empty_corpus(tmp_path: Path):
    c = OpenAIFineTuneClient(
        agent="jarvis",
        corpus_base=tmp_path / "tune",
        state_dir=tmp_path / "state",
        openai_client=MockOpenAI(),
    )
    res = c.trigger()
    assert res["status"] == "skipped"
    assert "empty corpus" in res["reason"]


def test_trigger_skips_below_threshold(tmp_path: Path):
    _write_corpus(tmp_path / "tune" / "jarvis" / "corpus.jsonl", n=10)
    c = OpenAIFineTuneClient(
        agent="jarvis",
        corpus_base=tmp_path / "tune",
        state_dir=tmp_path / "state",
        min_new_examples=50,
        openai_client=MockOpenAI(),
    )
    res = c.trigger()
    assert res["status"] == "skipped"
    assert "10 new examples" in res["reason"]


def test_trigger_starts_job_when_threshold_met(tmp_path: Path):
    _write_corpus(tmp_path / "tune" / "jarvis" / "corpus.jsonl", n=60)
    mock = MockOpenAI()
    c = OpenAIFineTuneClient(
        agent="jarvis",
        corpus_base=tmp_path / "tune",
        state_dir=tmp_path / "state",
        min_new_examples=50,
        openai_client=mock,
    )
    res = c.trigger()
    assert res["status"] == "started"
    assert res["examples"] == 60
    assert res["delta"] == 60
    assert mock.uploads_called == 1
    assert mock.creates_called == 1
    # State persists
    state = json.loads((tmp_path / "state" / "jarvis" / "finetune-state.json").read_text())
    assert state["last_job_id"] == res["job_id"]
    assert state["last_examples"] == 60
    assert len(state["history"]) == 1


def test_trigger_skips_when_previous_job_running(tmp_path: Path):
    _write_corpus(tmp_path / "tune" / "jarvis" / "corpus.jsonl", n=60)
    mock = MockOpenAI()
    c = OpenAIFineTuneClient(
        agent="jarvis",
        corpus_base=tmp_path / "tune",
        state_dir=tmp_path / "state",
        min_new_examples=50,
        openai_client=mock,
    )
    # First trigger: starts a job
    c.trigger()
    # Second trigger immediately: job is still in validating_files state per state file
    res2 = c.trigger()
    assert res2["status"] == "skipped"
    assert "still" in res2["reason"]
    # Mock should only have been called once for create
    assert mock.creates_called == 1


def test_status_captures_model_id_when_succeeded(tmp_path: Path):
    _write_corpus(tmp_path / "tune" / "jarvis" / "corpus.jsonl", n=60)
    mock = MockOpenAI()
    c = OpenAIFineTuneClient(
        agent="jarvis",
        corpus_base=tmp_path / "tune",
        state_dir=tmp_path / "state",
        min_new_examples=50,
        openai_client=mock,
    )
    c.trigger()
    # Now simulate the job finishing
    mock.next_status = "succeeded"
    mock.next_model = "ft:gpt-4o-mini-2024-07-18:org::personalized-jarvis-001"
    res = c.status()
    assert res["status"] == "succeeded"
    assert res["current_model"] == "ft:gpt-4o-mini-2024-07-18:org::personalized-jarvis-001"
    state = json.loads((tmp_path / "state" / "jarvis" / "finetune-state.json").read_text())
    assert state["current_model"] == res["current_model"]
    assert state["history"][-1]["model"] == res["current_model"]
    assert state["history"][-1]["status"] == "succeeded"


def test_status_no_job_yet(tmp_path: Path):
    c = OpenAIFineTuneClient(
        agent="jarvis",
        corpus_base=tmp_path / "tune",
        state_dir=tmp_path / "state",
        openai_client=MockOpenAI(),
    )
    res = c.status()
    assert res["status"] == "no-job"
    assert res["current_model"] is None


def test_status_failed_job_does_not_overwrite_current_model(tmp_path: Path):
    """If a previous fine-tune already produced a current_model, a NEW failed
    job should not blow that out. (Only succeeded jobs update current_model.)"""
    _write_corpus(tmp_path / "tune" / "jarvis" / "corpus.jsonl", n=60)
    mock = MockOpenAI()
    c = OpenAIFineTuneClient(
        agent="jarvis",
        corpus_base=tmp_path / "tune",
        state_dir=tmp_path / "state",
        min_new_examples=50,
        openai_client=mock,
    )
    # First succeeded job
    c.trigger()
    mock.next_status = "succeeded"
    mock.next_model = "ft:original"
    c.status()
    # Now grow corpus + trigger a 2nd job that fails
    _write_corpus(tmp_path / "tune" / "jarvis" / "corpus.jsonl", n=120)
    c.trigger()
    mock.next_status = "failed"
    mock.next_model = None
    res = c.status()
    assert res["status"] == "failed"
    # current_model from the FIRST success should still be present
    assert res["current_model"] == "ft:original"
