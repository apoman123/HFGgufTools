"""Convert HF model directories to GGUF and quantize the result."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .llama_cpp import LlamaCpp

# Precision types accepted by convert_hf_to_gguf.py's --outtype flag.
CONVERT_OUTTYPES = ["f32", "f16", "bf16", "q8_0", "auto"]


def _run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def convert_to_gguf(
    model_dir: Path,
    *,
    llama_cpp: LlamaCpp,
    out_file: Path,
    outtype: str = "f16",
    python: Path | None = None,
    extra_args: list[str] | None = None,
) -> Path:
    """Run ``convert_hf_to_gguf.py`` and return the produced GGUF path."""
    if outtype not in CONVERT_OUTTYPES:
        raise ValueError(
            f"Invalid outtype {outtype!r}; choose from {CONVERT_OUTTYPES}."
        )
    script = llama_cpp.convert_script
    if not script.is_file():
        raise FileNotFoundError(f"Converter script not found at {script}.")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(python or sys.executable),
        str(script),
        str(model_dir),
        "--outfile",
        str(out_file),
        "--outtype",
        outtype,
    ]
    if extra_args:
        cmd.extend(extra_args)
    _run(cmd)

    if not out_file.is_file():
        raise RuntimeError(
            f"Conversion reported success but {out_file} was not created."
        )
    return out_file


def quantize_gguf(
    in_file: Path,
    *,
    llama_cpp: LlamaCpp,
    quant_type: str,
    out_file: Path | None = None,
    allow_build: bool = True,
) -> Path:
    """Quantize a GGUF file with ``llama-quantize``; return the output path."""
    binary = llama_cpp.find_quantize_binary()
    if binary is None:
        if not allow_build:
            raise FileNotFoundError(
                "llama-quantize binary not found and building is disabled."
            )
        print("llama-quantize not found; building it now...", flush=True)
        binary = llama_cpp.build_quantize()

    if out_file is None:
        stem = in_file.stem
        # Replace a trailing precision tag (e.g. "-f16") with the quant type.
        for tag in ("-f16", "-f32", "-bf16", "-q8_0", "-auto"):
            if stem.endswith(tag):
                stem = stem[: -len(tag)]
                break
        out_file = in_file.with_name(f"{stem}-{quant_type}.gguf")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    _run([str(binary), str(in_file), str(out_file), quant_type])

    if not out_file.is_file():
        raise RuntimeError(
            f"Quantization reported success but {out_file} was not created."
        )
    return out_file
