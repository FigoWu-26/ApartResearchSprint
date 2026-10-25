"""Live-run names for the Modal vLLM servers in ``serve_vllm.py``.

Each alias is a different web endpoint. A live run must call that endpoint's URL,
not the Qwen server with another model name.
"""
from __future__ import annotations

import os

# Keys are the dashboard / CLI names. ``repo`` is the id vLLM serves.
# ``url_env`` is the compute.env variable for that server's OpenAI base URL (including /v1).
MODAL_MODELS: dict[str, dict] = {
    "qwen3-32b": {
        "label": "Qwen3 32B",
        "repo": "Qwen/Qwen3-32B-FP8",
        "url_env": "MODAL_LLM_BASE_URL",
        "disable_thinking": True,
    },
    "kimi": {
        "label": "Kimi VL A3B",
        "repo": "moonshotai/Kimi-VL-A3B-Instruct",
        "url_env": "MODAL_KIMI_BASE_URL",
        "disable_thinking": False,
    },
    "gemma": {
        "label": "Gemma 3 27B",
        "repo": "google/gemma-3-27b-it",
        "url_env": "MODAL_GEMMA_BASE_URL",
        "disable_thinking": False,
    },
}


def spec_for(name: str) -> dict | None:
    """Match an alias (``kimi``) or a served repo id, with or without a ``modal:`` prefix."""
    raw = str(name or "").strip()
    if raw.startswith("modal:"):
        raw = raw[len("modal:"):]
    if raw in MODAL_MODELS:
        return MODAL_MODELS[raw]
    for spec in MODAL_MODELS.values():
        if spec["repo"] == raw:
            return spec
    return None


def url_env_for(name: str) -> str:
    spec = spec_for(name)
    return spec["url_env"] if spec else "MODAL_LLM_BASE_URL"


def disable_thinking(name: str) -> bool:
    """Qwen3 emits ``<think>`` unless the request turns it off. The other servers do not take that flag."""
    spec = spec_for(name)
    if spec is None:
        return True
    return bool(spec["disable_thinking"])


def missing_modal_env(model_names, getenv=os.getenv) -> list[str]:
    """Env vars still required before a live Modal run may start. Empty means the selected servers are configured."""
    missing: list[str] = []
    if not (getenv("MODAL_LLM_API_KEY") or "").strip():
        missing.append("MODAL_LLM_API_KEY")
    seen: set[str] = set()
    for name in model_names:
        env = url_env_for(name)
        if env in seen:
            continue
        seen.add(env)
        if not (getenv(env) or "").strip():
            missing.append(env)
    return missing


def status_rows(getenv=os.getenv) -> list[dict]:
    key = bool((getenv("MODAL_LLM_API_KEY") or "").strip())
    return [
        {
            "id": name,
            "label": spec["label"],
            "ready": key and bool((getenv(spec["url_env"]) or "").strip()),
        }
        for name, spec in MODAL_MODELS.items()
    ]
