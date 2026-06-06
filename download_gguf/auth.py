"""Persisting and inspecting the HuggingFace access token.

These wrap ``huggingface_hub``'s own credential store (``~/.cache/huggingface/
token`` by default), so a token saved here is picked up automatically by every
later ``download`` run as well as by other HF tools on the machine.
"""

from __future__ import annotations

import os


def _resolve_token(token: str | None) -> str | None:
    """Token from the argument, then the standard HF env vars."""
    return (
        token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )


def login(token: str | None = None) -> str:
    """Persist an HF token and return the logged-in username.

    The token is taken from ``token``, then ``HF_TOKEN`` /
    ``HUGGING_FACE_HUB_TOKEN``, then an interactive prompt (hidden input).
    """
    from huggingface_hub import HfApi, login as hf_login

    token = _resolve_token(token)
    if not token:
        import getpass

        token = getpass.getpass(
            "HuggingFace token (https://huggingface.co/settings/tokens): "
        ).strip()
    if not token:
        raise ValueError("no token provided")

    # add_to_git_credential is best-effort; never let it abort the login.
    hf_login(token=token, add_to_git_credential=False)
    return HfApi().whoami(token=token)["name"]


def logout() -> None:
    """Remove any saved HF token."""
    from huggingface_hub import logout as hf_logout

    hf_logout()


def whoami() -> str | None:
    """Return the username for the saved/ambient token, or None if not logged in."""
    from huggingface_hub import HfApi
    from huggingface_hub.utils import LocalTokenNotFoundError

    try:
        return HfApi().whoami()["name"]
    except LocalTokenNotFoundError:
        return None
    except Exception:
        # An invalid/expired token raises an HTTP error; treat as logged out.
        return None
