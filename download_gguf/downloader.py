"""Download model repositories from the HuggingFace Hub."""

from __future__ import annotations

import os
from pathlib import Path

# File patterns that are never needed for GGUF conversion. Skipping them avoids
# pulling redundant weight formats (e.g. a repo that ships both .bin and
# .safetensors) and large unrelated assets.
DEFAULT_IGNORE_PATTERNS = [
    "*.gguf",  # already-converted weights
    "*.pth",  # original (non-HF) checkpoints, usually duplicated by safetensors
    "*.onnx",
    "*.onnx_data",
    "*.msgpack",  # flax
    "*.h5",  # tensorflow
    "*.tflite",
    "coreml/*",
    "*.mlmodel",
]


def _has_safetensors(repo_id: str, token: str | None) -> bool:
    """Return True if the repo contains at least one .safetensors file."""
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo_id, token=token)
    return any(f.endswith(".safetensors") for f in files)


def download_model(
    repo_id: str,
    *,
    out_dir: Path,
    revision: str | None = None,
    token: str | None = None,
    prefer_safetensors: bool = True,
    fast: bool = True,
) -> Path:
    """Download a HuggingFace model repo and return the local directory.

    Parameters
    ----------
    repo_id:
        The HuggingFace repo, e.g. ``"meta-llama/Llama-3.2-1B"``.
    out_dir:
        Directory the snapshot is materialised into.
    revision:
        Branch, tag or commit SHA. Defaults to the repo's default branch.
    token:
        HF access token (for gated/private repos). Falls back to the
        ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` environment variables.
    prefer_safetensors:
        When the repo ships both ``.bin`` and ``.safetensors`` weights, skip the
        pickle ``.bin`` files. Silently ignored when no safetensors exist.
    fast:
        Enable Xet high-performance transfer for higher-throughput downloads.
    """
    from huggingface_hub import snapshot_download

    if fast:
        os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

    token = token or os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN"
    )

    ignore_patterns = list(DEFAULT_IGNORE_PATTERNS)
    if prefer_safetensors and _has_safetensors(repo_id, token):
        ignore_patterns.append("*.bin")

    out_dir.mkdir(parents=True, exist_ok=True)
    local_path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        token=token,
        local_dir=str(out_dir),
        ignore_patterns=ignore_patterns,
    )
    return Path(local_path)
