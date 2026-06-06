"""Locate or provision a llama.cpp checkout.

llama.cpp owns both the HF->GGUF converter (``convert_hf_to_gguf.py``) and the
quantizer (the ``llama-quantize`` binary). This module finds an existing
checkout or clones one into a cache directory, and can install the converter's
Python requirements on demand.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp.git"

# Candidate names/locations for the quantize binary across build setups.
_QUANTIZE_NAMES = ["llama-quantize", "quantize", "llama-quantize.exe"]
_BUILD_BIN_DIRS = ["build/bin", "build", "."]


def _cache_root() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return Path(base) / "download-gguf"


def default_cache_dir() -> Path:
    """Return the cache directory used for an auto-cloned llama.cpp."""
    return _cache_root() / "llama.cpp"


def default_convert_venv_dir() -> Path:
    """Return the directory for the isolated converter virtualenv."""
    return _cache_root() / "convert-venv"


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _has_module(python: Path | str, module: str) -> bool:
    return (
        subprocess.run(
            [str(python), "-c", f"import {module}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _pip_install_cmd(python: Path) -> list[str]:
    """Return a working installer command targeting ``python``.

    Prefers ``uv pip`` when uv is available (it works on any venv and is needed
    for uv-created venvs, which ship without pip). The
    ``--index-strategy unsafe-best-match`` flag is required because the
    converter requirements pull torch from extra PyTorch indexes, and uv's
    default single-index strategy otherwise rejects packages (e.g.
    transformers) that resolve to a different version on those indexes. Falls
    back to ``python -m pip`` when uv is absent.
    """
    if shutil.which("uv") is not None:
        return [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "--index-strategy",
            "unsafe-best-match",
        ]
    if _has_module(python, "pip"):
        return [str(python), "-m", "pip", "install"]
    raise RuntimeError(
        "Neither uv nor pip is available to install the converter "
        "requirements. Install uv or pip, or install the requirements "
        "manually."
    )


class LlamaCpp:
    """A resolved llama.cpp checkout."""

    def __init__(self, root: Path):
        self.root = root

    @property
    def convert_script(self) -> Path:
        return self.root / "convert_hf_to_gguf.py"

    @property
    def convert_requirements(self) -> Path:
        return (
            self.root
            / "requirements"
            / "requirements-convert_hf_to_gguf.txt"
        )

    def find_quantize_binary(self) -> Path | None:
        for bin_dir in _BUILD_BIN_DIRS:
            for name in _QUANTIZE_NAMES:
                candidate = self.root / bin_dir / name
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    return candidate
        return None

    def converter_python(self, *, isolated: bool = True) -> Path:
        """Return the interpreter used to run the converter.

        When ``isolated`` (the default), a dedicated venv is created next to the
        llama.cpp checkout so the heavy converter dependencies (torch,
        transformers, ...) never clash with this tool's own runtime deps.
        Otherwise the current interpreter is returned.
        """
        if not isolated:
            return Path(sys.executable)

        venv_dir = default_convert_venv_dir()
        python = _venv_python(venv_dir)
        if python.is_file():
            return python

        venv_dir.parent.mkdir(parents=True, exist_ok=True)
        if shutil.which("uv") is not None:
            _run(["uv", "venv", str(venv_dir)])
        else:
            _run([sys.executable, "-m", "venv", str(venv_dir)])
        if not python.is_file():
            raise RuntimeError(f"Failed to create converter venv at {venv_dir}.")
        return python

    def converter_ready(self, python: Path) -> bool:
        """True if the converter deps are importable by ``python``."""
        return _has_module(python, "gguf") and _has_module(python, "transformers")

    def ensure_convert_requirements(self, python: Path) -> None:
        """Install the converter's Python dependencies into ``python``'s env."""
        req = self.convert_requirements
        if not req.is_file():
            raise FileNotFoundError(
                f"Converter requirements not found at {req}. "
                "Is this a valid llama.cpp checkout?"
            )
        _run([*_pip_install_cmd(python), "-r", str(req)])

    def build_quantize(self) -> Path:
        """Build the quantize binary with cmake; return its path."""
        if shutil.which("cmake") is None:
            raise RuntimeError(
                "cmake is required to build llama-quantize but was not found. "
                "Install cmake (and a C/C++ compiler), or pass a llama.cpp "
                "directory that already contains a built binary."
            )
        build_dir = self.root / "build"
        _run(["cmake", "-B", str(build_dir), "-DLLAMA_CURL=OFF"], cwd=self.root)
        _run(
            [
                "cmake",
                "--build",
                str(build_dir),
                "--config",
                "Release",
                "--target",
                "llama-quantize",
                "-j",
            ],
            cwd=self.root,
        )
        binary = self.find_quantize_binary()
        if binary is None:
            raise RuntimeError(
                "Build finished but no llama-quantize binary was found."
            )
        return binary


def resolve_llama_cpp(
    explicit_dir: Path | None = None,
    *,
    auto_clone: bool = True,
    update: bool = False,
) -> LlamaCpp:
    """Return a usable :class:`LlamaCpp`, cloning into the cache if needed."""
    if explicit_dir is not None:
        root = explicit_dir.expanduser().resolve()
        if not (root / "convert_hf_to_gguf.py").is_file():
            raise FileNotFoundError(
                f"{root} does not look like a llama.cpp checkout "
                "(missing convert_hf_to_gguf.py)."
            )
        return LlamaCpp(root)

    root = default_cache_dir()
    if (root / "convert_hf_to_gguf.py").is_file():
        if update:
            _run(["git", "-C", str(root), "pull", "--ff-only"])
        return LlamaCpp(root)

    if not auto_clone:
        raise FileNotFoundError(
            "No llama.cpp checkout found. Pass --llama-cpp-dir or allow "
            "auto-clone."
        )

    if shutil.which("git") is None:
        raise RuntimeError("git is required to clone llama.cpp but was not found.")

    root.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--depth", "1", LLAMA_CPP_REPO, str(root)])
    return LlamaCpp(root)
