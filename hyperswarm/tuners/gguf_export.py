"""GGUF export — fuse a trained LoRA adapter into the base model and write
a GGUF file that Ollama (and any other llama.cpp-based runner) can load.

Two-step pipeline:

  1. `mlx_lm fuse` (no --export-gguf): merges the LoRA adapter weights into
     the base model and writes a standard HuggingFace safetensors directory.
     Works for every model architecture mlx_lm itself supports (Llama,
     Mistral, Qwen2, Qwen3, Phi, etc.).

  2. `llama.cpp convert_hf_to_gguf.py`: converts the HF dir to GGUF. Handles
     all the architectures llama.cpp supports — broader than mlx_lm's
     built-in `--export-gguf` flag, which only knows Llama/Mistral families.

Why two steps instead of `mlx_lm fuse --export-gguf` directly: that flag
crashes with `Model type qwen3 not supported for GGUF conversion` (and same
for any newer architecture). The two-step approach is universal.

llama.cpp's converter is a single Python script. `pip install
llama-cpp-python` brings it along, OR `git clone https://github.com/ggerganov/llama.cpp`
gets you both the converter and the `quantize` tool for further size reduction.

State file (`finetune-state.json`) is the single source of truth for "which
model is the current personalized one." Updating `current_gguf` means
downstream routers (openclaw config swap, ollama Modelfile generation)
read one file to know what to load.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

DEFAULT_STATE_DIR = "~/.local/state/hyperswarm/tune"
DEFAULT_OUTPUT_BASE = "~/.openclaw/tune"


def _expand(p: str | os.PathLike) -> Path:
    return Path(os.path.expanduser(str(p)))


@dataclass
class GGUFExporter:
    agent: str
    base_model: str | None = None  # None → pull from state file
    state_dir: Path | None = None
    output_base: Path | None = None
    python_exec: str = sys.executable

    # Test/inject hook
    runner: callable | None = None

    def __post_init__(self) -> None:
        self.state_dir = _expand(self.state_dir or DEFAULT_STATE_DIR)
        self.output_base = _expand(self.output_base or DEFAULT_OUTPUT_BASE)

    @property
    def state_path(self) -> Path:
        return self.state_dir / self.agent / "finetune-state.json"

    def _load_state(self) -> dict | None:
        if not self.state_path.exists():
            return None
        try:
            return json.loads(self.state_path.read_text())
        except (ValueError, json.JSONDecodeError):
            return None

    def _save_state(self, state: dict) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2))

    def _real_run(self, *, cmd: list[str], cwd: Path, log_path: Path, append: bool = False) -> int:
        """Default runner. Test runners can ignore `append` (any extra kwargs)."""
        log_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with open(log_path, mode) as logf:
            logf.write(f"\n$ {' '.join(cmd)}\n\n")
            logf.flush()
            r = subprocess.run(
                cmd, cwd=str(cwd), stdout=logf, stderr=subprocess.STDOUT, check=False
            )
        return r.returncode

    def _call_runner(self, *, cmd: list[str], cwd: Path, log_path: Path, append: bool) -> int:
        """Invoke the configured runner, dropping `append` if it doesn't take it.
        Lets injected test runners stay tiny — `def runner(*, cmd, cwd, log_path)`."""
        runner = self.runner or self._real_run
        try:
            return runner(cmd=cmd, cwd=cwd, log_path=log_path, append=append)
        except TypeError:
            return runner(cmd=cmd, cwd=cwd, log_path=log_path)

    def _find_llama_cpp_converter(self) -> str | None:
        """Locate llama.cpp's convert_hf_to_gguf.py. Search order:
        1. $HYPERSWARM_LLAMA_CPP_DIR env var
        2. ~/llama.cpp (a common manual clone location)
        3. ~/projects/llama.cpp (Mac convention)
        4. /opt/llama.cpp (Linux convention)
        Returns the script path, or None if not found.
        """
        candidates = []
        env = os.environ.get("HYPERSWARM_LLAMA_CPP_DIR")
        if env:
            candidates.append(Path(env))
        candidates += [
            Path.home() / "llama.cpp",
            Path.home() / "projects" / "llama.cpp",
            Path("/opt/llama.cpp"),
        ]
        for d in candidates:
            for name in ("convert_hf_to_gguf.py", "convert-hf-to-gguf.py"):
                p = d / name
                if p.exists():
                    return str(p)
        return None

    def export(self) -> dict:
        """Two-step: fuse adapter into HF safetensors (`mlx_lm fuse --dequantize`),
        then convert HF → GGUF (`llama.cpp/convert_hf_to_gguf.py`). Records
        `current_gguf` in state on success.

        Idempotency: if the fused dir already has safetensors, step 1 is
        skipped (lets you re-run after installing llama.cpp without re-fusing).
        """
        state = self._load_state()
        if not state:
            return {"status": "failed", "reason": "no finetune-state.json — train an adapter first"}
        adapter_path = state.get("current_adapter")
        if not adapter_path:
            return {"status": "failed", "reason": "no current_adapter in state — train an adapter first"}
        adapter_dir = Path(adapter_path)
        if not adapter_dir.exists():
            return {"status": "failed", "reason": f"current_adapter does not exist on disk: {adapter_dir}"}

        base_model = self.base_model or state.get("model_base")
        if not base_model:
            return {"status": "failed", "reason": "no base_model — pass --base-model or train first"}

        run_dir = adapter_dir.parent
        fused_dir = run_dir / "fused"
        gguf_path = run_dir / f"{Path(base_model).name}-merged-f16.gguf"
        log_path = run_dir / "fuse.log"

        # ── Step 1: mlx_lm fuse (no --export-gguf) → HF safetensors dir ────
        # `--dequantize` is required when the base model is 4-bit
        # (mlx-community/<...>-4bit) so llama.cpp's converter doesn't choke on
        # leftover quantization tensors (lm_head.biases / .scales). For
        # non-quantized bases it's a no-op.
        if not (fused_dir.exists() and any(fused_dir.glob("*.safetensors"))):
            fuse_cmd = [
                self.python_exec, "-m", "mlx_lm", "fuse",
                "--model", base_model,
                "--adapter-path", str(adapter_dir),
                "--save-path", str(fused_dir),
                "--dequantize",
            ]
            try:
                rc = self._call_runner(cmd=fuse_cmd, cwd=run_dir, log_path=log_path, append=False)
            except Exception as e:
                return {"status": "failed", "reason": f"fuse subprocess crashed: {e}"}
            if rc != 0:
                return {
                    "status": "failed",
                    "exit_code": rc,
                    "log_path": str(log_path),
                    "reason": "mlx_lm fuse returned non-zero; see log",
                }
            if not any(fused_dir.glob("*.safetensors")):
                return {
                    "status": "failed",
                    "reason": "fuse exited 0 but no safetensors written under fused/",
                }

        # ── Step 2: llama.cpp convert_hf_to_gguf.py → GGUF ──────────────────
        converter = self._find_llama_cpp_converter()
        if converter is None:
            return {
                "status": "failed",
                "reason": (
                    "llama.cpp converter not found. Step 1 (fuse) succeeded — "
                    "the fused HF model is at "
                    f"{fused_dir}. To finish, install llama.cpp and re-run:\n"
                    "  git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp\n"
                    "  pip install -r ~/llama.cpp/requirements.txt\n"
                    "Or set HYPERSWARM_LLAMA_CPP_DIR to a custom location."
                ),
                "fused_dir": str(fused_dir),
                "step_1_completed": True,
            }
        convert_cmd = [
            self.python_exec, converter,
            str(fused_dir),
            "--outfile", str(gguf_path),
            "--outtype", "f16",
        ]
        try:
            rc = self._call_runner(cmd=convert_cmd, cwd=run_dir, log_path=log_path, append=True)
        except Exception as e:
            return {
                "status": "failed",
                "reason": f"llama.cpp convert subprocess crashed: {e}",
                "fused_dir": str(fused_dir),
                "step_1_completed": True,
            }
        if rc != 0:
            return {
                "status": "failed",
                "exit_code": rc,
                "log_path": str(log_path),
                "reason": "llama.cpp convert returned non-zero; see log",
                "step_1_completed": True,
            }
        if not gguf_path.exists():
            return {
                "status": "failed",
                "reason": f"convert exited 0 but {gguf_path} was not written",
                "step_1_completed": True,
            }

        # ── Record in state ─────────────────────────────────────────────────
        state["current_gguf"] = str(gguf_path)
        state.setdefault("gguf_history", []).append(
            {
                "gguf_path": str(gguf_path),
                "fused_from_adapter": str(adapter_dir),
                "fused_dir": str(fused_dir),
                "base_model": base_model,
                "converter": converter,
                "completed_at": int(time.time()),
            }
        )
        self._save_state(state)
        return {
            "status": "completed",
            "gguf_path": str(gguf_path),
            "fused_dir": str(fused_dir),
            "converter": converter,
            "size_bytes": gguf_path.stat().st_size,
            "log_path": str(log_path),
        }


def export_gguf(agent: str, **kwargs) -> dict:
    return GGUFExporter(agent=agent, **kwargs).export()
