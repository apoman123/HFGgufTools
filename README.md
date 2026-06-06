# download-gguf

Download a model from the HuggingFace Hub and convert it to **GGUF**, convert a
**local** model directory to GGUF, or quantize an existing **local** GGUF file.

```
download   HuggingFace repo -> download weights -> convert to GGUF -> (optional) quantize
convert    local model dir  ->                     convert to GGUF -> (optional) quantize
quantize   local .gguf file ->                     quantize
login      save a HuggingFace token (for gated/private repos)
logout     remove the saved token
whoami     show the user for the saved token
```

The heavy lifting for conversion and quantization is done by
[llama.cpp](https://github.com/ggml-org/llama.cpp): this tool wraps its
`convert_hf_to_gguf.py` script and `llama-quantize` binary, locating an existing
checkout or cloning one into `~/.cache/download-gguf/llama.cpp` automatically.

The converter's heavy dependencies (PyTorch, Transformers, ...) are installed
into a **dedicated virtualenv** at `~/.cache/download-gguf/convert-venv` on first
use, so they never clash with this tool's own runtime deps. You don't need to
install them yourself.

## Install

```bash
uv sync
```

That's it — the lightweight runtime deps (`huggingface-hub` + Xet) are all you
need locally. Converter deps are provisioned automatically on first conversion.

## Usage

### `download` — from HuggingFace

```bash
# Download + convert to f16 GGUF
uv run download-gguf download meta-llama/Llama-3.2-1B

# Download, convert, then quantize to Q4_K_M (intermediate f16 removed)
uv run download-gguf download meta-llama/Llama-3.2-1B --quantize Q4_K_M

# Private/gated repo at a specific revision
uv run download-gguf download org/Model --token "$HF_TOKEN" --revision main

# Just download, don't convert
uv run download-gguf download org/Model --download-only

# Put the downloaded weights and the GGUF in separate directories
uv run download-gguf download org/Model \
  --weights-dir ./hf_weights \
  --gguf-dir    ./gguf_out
# -> ./hf_weights/Model/ (safetensors, ...)   ./gguf_out/Model-f16.gguf
```

### `login` — save a HuggingFace token

For gated or private repos you can save a token once instead of passing
`--token` on every `download`. The token is stored in the standard HF
credential store (`~/.cache/huggingface/token`) and is picked up automatically
by later downloads (and other HF tools).

```bash
# Interactive (hidden) prompt
uv run download-gguf login

# Non-interactive (also reads HF_TOKEN / HUGGING_FACE_HUB_TOKEN if --token omitted)
uv run download-gguf login --token "$HF_TOKEN"

uv run download-gguf whoami    # show the logged-in user
uv run download-gguf logout    # remove the saved token
```

### `convert` — a local model directory

```bash
# Convert an already-downloaded HF model dir to GGUF
uv run download-gguf convert ./models/Llama-3.2-1B

# Convert and quantize in one go
uv run download-gguf convert /path/to/model --outtype bf16 --quantize Q5_K_M
```

### `quantize` — a local GGUF file

```bash
# Quantize an existing GGUF; output is written next to the input
uv run download-gguf quantize ./model-f16.gguf Q4_K_M

# Choose an explicit output path
uv run download-gguf quantize ./model-f16.gguf Q4_K_M -o ./model-q4.gguf
```

Any type accepted by `llama-quantize` works. Common choices:

| Type | Notes |
|------|-------|
| `Q4_K_M` | Good size/quality balance — the usual default. |
| `Q5_K_M` | Larger, a bit higher quality. |
| `Q8_0` | Near-lossless, ~2× the size of a 4-bit quant. |
| `Q3_K_M` / `Q2_K` | Smallest, with more quality loss. |

Outputs land under `models/<model-name>/` by default (`-o/--outdir` to change).

## Options by subcommand

**`download`** — `repo_id` plus:

| Flag | Purpose |
|------|---------|
| `--revision` / `--token` | HF branch/tag/SHA and access token. |
| `--no-fast` | Disable Xet accelerated downloads. |
| `--keep-bin` | Download `.bin` weights even when safetensors exist. |
| `--weights-dir PATH` | Directory for the downloaded weights (default: `<outdir>`); weights go in `<weights-dir>/<name>/`. |
| `--download-only` | Stop after downloading. |
| *(conversion + llama.cpp options below)* | |

**`convert`** — `model_dir` plus the conversion and llama.cpp options below.

**`quantize`** — `gguf_file` and `quant_type` (e.g. `Q4_K_M`) plus:

| Flag | Purpose |
|------|---------|
| `-o, --outfile PATH` | Output path (default: next to input, named by quant type). |
| `--no-build` | Fail instead of building `llama-quantize` when no binary exists. |
| *(llama.cpp options below)* | |

**Conversion options** (`download`, `convert`):

| Flag | Purpose |
|------|---------|
| `--outtype {f32,f16,bf16,q8_0,auto}` | GGUF precision from the converter (default `f16`). |
| `-q, --quantize Q4_K_M` | Quantize the GGUF afterwards (any `llama-quantize` type). |
| `--keep-intermediate` | Keep the unquantized GGUF when quantizing. |
| `-o, --outdir` / `--name` | Base output directory and base file name. |
| `--gguf-dir PATH` | Directory for the output GGUF file (default: `<outdir>/<name>`). |
| `--install-deps` | Force a reinstall of the converter requirements. |
| `--no-isolation` | Run the converter in the current env (bring your own deps). |

**llama.cpp options** (all subcommands):

| Flag | Purpose |
|------|---------|
| `--llama-cpp-dir PATH` | Use an existing llama.cpp checkout. |
| `--update-llama-cpp` | `git pull` the cached checkout before use. |
| `--no-auto-clone` | Error instead of cloning llama.cpp. |

`--token` falls back to the `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` env vars.

## Requirements

- Python ≥ 3.12
- `git` (to clone llama.cpp when not supplied)
- `cmake` + a C/C++ compiler — only needed if `--quantize` is used and no
  prebuilt `llama-quantize` binary exists in the llama.cpp checkout.

## Caches & disk usage

On first use the tool populates `~/.cache/download-gguf/`:

| Path | Contents |
|------|----------|
| `llama.cpp/` | Auto-cloned llama.cpp checkout (converter + quantizer source). |
| `convert-venv/` | Isolated virtualenv holding the converter deps — **includes PyTorch, so it can be several GB.** |

Both are reused across runs. Delete `~/.cache/download-gguf/` to reclaim the
space (it will be recreated on the next conversion). Use `--llama-cpp-dir` to
point at your own checkout instead of the cached one, and `--no-isolation` to
skip the dedicated venv and use the current environment's packages.
