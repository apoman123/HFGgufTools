"""Command-line entry point for download_gguf.

Three subcommands:

* ``download`` - pull a model from the HuggingFace Hub, convert it to GGUF and
  optionally quantize the result.
* ``convert``  - convert an already-downloaded local model directory to GGUF
  (optionally quantizing).
* ``quantize`` - quantize an existing local GGUF file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .converter import CONVERT_OUTTYPES, convert_to_gguf, quantize_gguf
from .downloader import download_model
from .llama_cpp import resolve_llama_cpp


def _slug(repo_id: str) -> str:
    """Turn 'org/Model-Name' into a filesystem-friendly 'Model-Name'."""
    return repo_id.rstrip("/").split("/")[-1]


# --------------------------------------------------------------------------- #
# Shared argument groups
# --------------------------------------------------------------------------- #
def _add_output_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("output")
    g.add_argument(
        "-o", "--outdir", type=Path, default=Path("models"),
        help="Base output directory for downloads and GGUF files.",
    )
    g.add_argument(
        "--gguf-dir", type=Path, default=None,
        help="Directory for the output GGUF file (default: <outdir>/<name>).",
    )
    g.add_argument(
        "--name", default=None,
        help="Base name for output files (defaults to the model name).",
    )


def _add_convert_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("conversion")
    g.add_argument(
        "--outtype", choices=CONVERT_OUTTYPES, default="f16",
        help="GGUF precision produced by the converter.",
    )
    g.add_argument(
        "--quantize", "-q", default=None,
        help="Quantization type to apply after conversion, e.g. 'Q4_K_M'.",
    )
    g.add_argument(
        "--keep-intermediate", action="store_true",
        help="Keep the unquantized GGUF when --quantize is used.",
    )


def _add_llama_cpp_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("llama.cpp")
    g.add_argument(
        "--llama-cpp-dir", type=Path, default=None,
        help="Path to an existing llama.cpp checkout (otherwise auto-cloned).",
    )
    g.add_argument(
        "--update-llama-cpp", action="store_true",
        help="git pull the cached llama.cpp checkout before use.",
    )
    g.add_argument(
        "--no-auto-clone", action="store_true",
        help="Fail instead of cloning llama.cpp when none is found.",
    )


def _add_converter_env_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("converter environment")
    g.add_argument(
        "--install-deps", action="store_true",
        help="Force (re)install of llama.cpp's converter requirements.",
    )
    g.add_argument(
        "--no-isolation", action="store_true",
        help=(
            "Run the converter in the current environment instead of a "
            "dedicated venv (you must supply torch/transformers/gguf yourself)."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="download-gguf",
        description=(
            "Download HuggingFace models, convert them to GGUF, and quantize "
            "GGUF files."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", metavar="{download,convert,quantize}")
    sub.required = True

    # -- download -------------------------------------------------------- #
    dl = sub.add_parser(
        "download",
        help="Download a HF model, convert to GGUF, optionally quantize.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    dl.add_argument(
        "repo_id", help="HuggingFace repo id, e.g. 'meta-llama/Llama-3.2-1B'."
    )
    g = dl.add_argument_group("download")
    g.add_argument("--revision", default=None, help="Branch, tag, or commit SHA.")
    g.add_argument(
        "--token", default=None, help="HF access token (gated/private repos)."
    )
    g.add_argument(
        "--no-fast", action="store_true",
        help="Disable Xet accelerated downloads.",
    )
    g.add_argument(
        "--keep-bin", action="store_true",
        help="Download .bin weights even when safetensors are available.",
    )
    g.add_argument(
        "--weights-dir", type=Path, default=None,
        help=(
            "Directory for the downloaded weights (default: <outdir>). "
            "Weights are placed in <weights-dir>/<name>/."
        ),
    )
    g.add_argument(
        "--download-only", action="store_true",
        help="Only download the model; skip conversion.",
    )
    _add_output_args(dl)
    _add_convert_args(dl)
    _add_llama_cpp_args(dl)
    _add_converter_env_args(dl)
    dl.set_defaults(func=run_download)

    # -- convert --------------------------------------------------------- #
    cv = sub.add_parser(
        "convert",
        help="Convert a local model directory to GGUF, optionally quantize.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    cv.add_argument(
        "model_dir", type=Path,
        help="Path to a local HF model directory (containing config.json).",
    )
    _add_output_args(cv)
    _add_convert_args(cv)
    _add_llama_cpp_args(cv)
    _add_converter_env_args(cv)
    cv.set_defaults(func=run_convert)

    # -- quantize -------------------------------------------------------- #
    qz = sub.add_parser(
        "quantize",
        help="Quantize an existing local GGUF file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    qz.add_argument("gguf_file", type=Path, help="Path to the input .gguf file.")
    qz.add_argument(
        "quant_type", help="Quantization type, e.g. 'Q4_K_M', 'Q5_K_M', 'Q8_0'."
    )
    qz.add_argument(
        "-o", "--outfile", type=Path, default=None,
        help="Output path (default: alongside the input, named by quant type).",
    )
    qz.add_argument(
        "--no-build", action="store_true",
        help="Fail instead of building llama-quantize when no binary is found.",
    )
    _add_llama_cpp_args(qz)
    qz.set_defaults(func=run_quantize)

    return p


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _resolve_llama(args: argparse.Namespace):
    return resolve_llama_cpp(
        args.llama_cpp_dir,
        auto_clone=not args.no_auto_clone,
        update=args.update_llama_cpp,
    )


def _gguf_dir(args: argparse.Namespace, name: str) -> Path:
    """Resolve the directory the GGUF file is written to."""
    base = args.gguf_dir if args.gguf_dir is not None else args.outdir / name
    return base.expanduser().resolve()


def _convert_and_quantize(
    model_dir: Path, *, name: str, gguf_dir: Path, args: argparse.Namespace
) -> int:
    """Convert ``model_dir`` to GGUF and optionally quantize. Shared by the
    ``download`` and ``convert`` subcommands."""
    llama_cpp = _resolve_llama(args)

    # Prepare the interpreter that runs the converter. By default this is a
    # dedicated venv so the heavy converter deps don't clash with our own.
    convert_python = llama_cpp.converter_python(isolated=not args.no_isolation)
    if args.install_deps or not llama_cpp.converter_ready(convert_python):
        print("Installing converter requirements ...")
        llama_cpp.ensure_convert_requirements(convert_python)

    gguf_path = gguf_dir / f"{name}-{args.outtype}.gguf"
    print(f"Converting to GGUF ({args.outtype}) ...")
    gguf_path = convert_to_gguf(
        model_dir,
        llama_cpp=llama_cpp,
        out_file=gguf_path,
        outtype=args.outtype,
        python=convert_python,
    )
    print(f"GGUF written to {gguf_path}")

    if args.quantize:
        print(f"Quantizing to {args.quantize} ...")
        quant_path = quantize_gguf(
            gguf_path, llama_cpp=llama_cpp, quant_type=args.quantize
        )
        print(f"Quantized GGUF written to {quant_path}")
        if not args.keep_intermediate:
            gguf_path.unlink(missing_ok=True)
            print(f"Removed intermediate {gguf_path}")

    return 0


# --------------------------------------------------------------------------- #
# Subcommand entry points
# --------------------------------------------------------------------------- #
def run_download(args: argparse.Namespace) -> int:
    name = args.name or _slug(args.repo_id)
    weights_base = args.weights_dir if args.weights_dir is not None else args.outdir
    weights_dir = (weights_base / name).expanduser().resolve()

    print(f"Downloading {args.repo_id} ...")
    model_dir = download_model(
        args.repo_id,
        out_dir=weights_dir,
        revision=args.revision,
        token=args.token,
        prefer_safetensors=not args.keep_bin,
        fast=not args.no_fast,
    )
    print(f"Downloaded to {model_dir}")

    if args.download_only:
        return 0
    return _convert_and_quantize(
        model_dir, name=name, gguf_dir=_gguf_dir(args, name), args=args
    )


def run_convert(args: argparse.Namespace) -> int:
    model_dir = args.model_dir.expanduser().resolve()
    if not model_dir.is_dir():
        print(f"error: model directory {model_dir} does not exist", file=sys.stderr)
        return 1
    if not (model_dir / "config.json").is_file():
        print(
            f"error: {model_dir} has no config.json; it does not look like a "
            "HuggingFace model directory.",
            file=sys.stderr,
        )
        return 1

    name = args.name or model_dir.name
    return _convert_and_quantize(
        model_dir, name=name, gguf_dir=_gguf_dir(args, name), args=args
    )


def run_quantize(args: argparse.Namespace) -> int:
    in_file = args.gguf_file.expanduser().resolve()
    if not in_file.is_file():
        print(f"error: GGUF file {in_file} does not exist", file=sys.stderr)
        return 1
    if in_file.suffix.lower() != ".gguf":
        print(f"error: {in_file} does not have a .gguf extension", file=sys.stderr)
        return 1

    llama_cpp = _resolve_llama(args)
    out_file = args.outfile.expanduser().resolve() if args.outfile else None

    print(f"Quantizing {in_file.name} to {args.quant_type} ...")
    quant_path = quantize_gguf(
        in_file,
        llama_cpp=llama_cpp,
        quant_type=args.quant_type,
        out_file=out_file,
        allow_build=not args.no_build,
    )
    print(f"Quantized GGUF written to {quant_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:  # surface a clean message, not a traceback
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
