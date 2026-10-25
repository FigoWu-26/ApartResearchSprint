"""
LLM providers for the laptop-scale harness (run_game.py, branch_game.py, api_judge.py).

Model calls follow the shared inference config (poseidon_compute): **Nebius** by
default, with **OpenRouter** as the fallback for closed models
(``NEBIUS_API_KEY`` / optional ``OPENROUTER_API_KEY`` in compute.env). The paper ran
DeepSeek-R1 distills (7B/14B/32B) through vLLM on GPUs; those distills are no longer
served on OpenRouter, so the default is the full reasoning model
``deepseek-ai/DeepSeek-R1-0528``. Any model id works, written in Nebius / Hugging Face
casing (poseidon_compute translates to OpenRouter's casing when it routes there), e.g.
``Qwen/Qwen3-14B`` or ``openai/gpt-4.1-mini``.

Reasoning models return their chain of thought either inside ``<think>`` tags or
in OpenRouter's ``reasoning`` field; both are captured as the agent's "thoughts".

Two opt-in prefixes remain for running without the shared endpoint: ``ollama:`` (a free local
model) and ``custom:`` (any OpenAI-compatible endpoint via ``LLM_BASE_URL``). The former
``openai:``, ``google:`` and ``mistral:`` presets -- direct provider endpoints with their
own keys -- were removed; use the model id instead (``openai/gpt-4.1``,
``google/gemini-2.5-flash``, ``mistralai/...``).
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple



from poseidon_compute import make_client, resolve_model

try:  # endpoint-aware diagnostics (where to check model ids / rates); optional at import time
    from poseidon_compute import endpoint_kind
except Exception:  # noqa: BLE001
    endpoint_kind = None

Message = Dict[str, str]

# Model-catalog / rate pages per active endpoint. Printed in diagnostics so a Nebius run doesn't
# point at OpenRouter (or vice-versa). Local/custom endpoints have no public catalog.
_CATALOG_URLS = {
    "nebius": "https://studio.nebius.com/models",
    "openrouter": "https://openrouter.ai/models",
}


def _catalog_url() -> Optional[str]:
    """The model-catalog/rates page for whichever endpoint make_client() is using, or None."""
    kind = endpoint_kind() if endpoint_kind is not None else None
    return _CATALOG_URLS.get(kind)


@dataclass(frozen=True)
class Preset:
    base_url: str
    key_env: str
    default_model: str
    needs_key: bool = True
    supports_seed: bool = False


# The paper ran DeepSeek-R1-Distill-Qwen 7B/14B/32B via vLLM. run_game.py makes participants choose
# their agent model explicitly; this constant is only the fallback for bare `openrouter:` specs. Model
# ids are in Nebius / Hugging Face casing (used as-is; nothing rewritten).
PAPER_MODEL = "deepseek-ai/DeepSeek-R1-0528"
# The paper's judge was Qwen3-14B, run via vLLM with thinking OFF to keep answers to a one-line score.
# api_judge.py runs the judge with reasoning=False and max_tokens=64; OpenRouter CANNOT disable Qwen3's
# thinking, so a hybrid reasoning model spends the 64-token budget on <think> and never emits a score —
# judging fails. Default instead to a plain (non-reasoning) instruct model that returns "Score: N"
# directly on the shared endpoint. Override with --model to use a different judge.
JUDGE_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # non-reasoning instruct judge (works with reasoning=False)

PRESETS: Dict[str, Preset] = {
    # Hackathon default: one key, every model.
    "openrouter": Preset(
        "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", PAPER_MODEL, supports_seed=True
    ),
    # Local / offline escape hatch (no key)
    "ollama": Preset("http://localhost:11434/v1", "OLLAMA_API_KEY", "llama3.2:3b", needs_key=False),
    # Generic: LLM_BASE_URL + LLM_API_KEY
    "custom": Preset("", "LLM_API_KEY", "", needs_key=False),
}


_REMOVED_PREFIXES = {
    "openai": "openai/<model>",
    "google": "google/<model>",
    "mistral": "mistralai/<model>",
}


def parse_model_spec(spec: str) -> Tuple[str, str]:
    """'ollama:llama3.2:3b' -> ('ollama', 'llama3.2:3b'); bare name -> (LLM_PROVIDER, name)."""
    if ":" in spec:
        prefix, rest = spec.split(":", 1)
        if prefix in PRESETS:
            return prefix, rest
        if prefix in _REMOVED_PREFIXES:
            raise ValueError(
                f"The '{prefix}:' prefix was removed: every hosted model goes through the shared inference config. "
                f"Use the model id instead (Nebius / Hugging Face casing), e.g. {_REMOVED_PREFIXES[prefix].replace('<model>', rest)}."
            )
    return os.getenv("LLM_PROVIDER", "openrouter"), spec


class FatalProviderError(RuntimeError):
    """Model not found / bad request / bad key: retrying cannot help, abort the run."""


class ChatProvider:
    """One OpenAI-compatible chat-completions client for a (provider, model) pair."""

    def __init__(self, spec: str, temperature: float = 1.0, max_tokens: int = 4096, reasoning: bool = True):
        """reasoning=False asks OpenRouter to disable thinking (used by the judge to keep answers short)."""
        self.provider, self.model = parse_model_spec(spec)
        preset = PRESETS[self.provider]
        self.base_url = os.getenv("LLM_BASE_URL") if self.provider == "custom" else preset.base_url
        if not self.base_url:
            raise ValueError("LLM_BASE_URL must be set for provider 'custom'")
        self.api_key = os.getenv(preset.key_env)
        # resolve_model: apply retired-pin upgrades, LLM_MODEL override and OpenRouter casing to
        # BOTH the explicit id and the preset default, so an explicit --models id gets the same
        # handling as the default (not passed through verbatim).
        self.model = resolve_model(self.model or preset.default_model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.supports_seed = preset.supports_seed
        self.is_local = "localhost" in self.base_url or "127.0.0.1" in self.base_url

        self.reasoning = reasoning

        if self.provider == "openrouter":
            # Default preset: route via the shared compute.env config (Nebius or OpenRouter, with the
            # OpenRouter fallback) — make_client() picks endpoint + base_url + key.
            self._client = make_client()
        else:
            # Non-default preset (ollama / custom): honor the preset's OWN base_url/key, otherwise a
            # chosen non-default provider would silently route to the shared endpoint. A keyless
            # endpoint (ollama) still needs a non-empty placeholder for the OpenAI client.
            from openai import OpenAI

            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key or "EMPTY")

    @property
    def spec(self) -> str:
        return f"{self.provider}:{self.model}"

    def complete(self, messages: List[Message], seed: Optional[int] = None) -> Tuple[str, dict]:
        """Return (text, metrics). Raises on transport errors."""
        start = time.time()
        kwargs = {}
        if seed is not None and self.supports_seed:
            kwargs["seed"] = seed
        if self.provider == "openrouter" and not self.reasoning:
            kwargs["extra_body"] = {"reasoning": {"enabled": False}}
        from openai import AuthenticationError, BadRequestError, NotFoundError, PermissionDeniedError

        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **kwargs,
            )
        except (NotFoundError, AuthenticationError, PermissionDeniedError, BadRequestError) as e:
            catalog = _catalog_url()
            where = f" Check the model id at {catalog} and your key." if catalog else " Check the model id and your key."
            raise FatalProviderError(f"{self.spec}: {e}.{where}") from e
        if not resp.choices:
            raise RuntimeError(f"{self.spec}: empty response ({getattr(resp, 'error', None)})")
        msg = resp.choices[0].message
        text = msg.content or ""
        # OpenRouter puts reasoning-model thoughts in a separate field; fold them in as <think>
        extra = getattr(msg, "model_extra", None) or {}
        reasoning = extra.get("reasoning") or getattr(msg, "reasoning", None)
        if reasoning and "</think>" not in text:
            text = f"<think>{reasoning}</think>\n{text}"
        usage = getattr(resp, "usage", None)
        metrics = {
            "model": self.spec,
            "temperature": self.temperature,
            "latency_seconds": round(time.time() - start, 3),
            "tokens_input": getattr(usage, "prompt_tokens", None),
            "tokens_output": getattr(usage, "completion_tokens", None),
        }
        return text, metrics


_PRICE_PATTERNS = [
    r"chosen\s*price\s*[:=]\s*\**\s*\$?\s*(-?\d+(?:\.\d+)?)",
    r"price\s*[:=]\s*\**\s*\$?\s*(-?\d+(?:\.\d+)?)",
]


def split_thoughts_and_price(text: str) -> Tuple[str, Optional[float]]:
    """
    Parse 'My chosen price: 1.62'. A <think>...</think> block (reasoning models)
    plus anything before the *last* 'chosen price' line is the agent's 'thoughts'.
    """
    text = text.strip()
    think = ""
    if "</think>" in text:
        think, text = text.rsplit("</think>", 1)
        think = think.replace("<think>", "").strip()
        text = text.strip()
    matches = list(re.finditer(r"(my\s+)?chosen\s*price", text, flags=re.IGNORECASE))
    idx = matches[-1].start() if matches else -1
    thoughts = text[:idx].strip() if idx > 0 else ""
    thoughts = "\n".join(t for t in (think, thoughts) if t)
    tail = text[idx:] if idx >= 0 else text
    for pat in _PRICE_PATTERNS:
        m = re.search(pat, tail, flags=re.IGNORECASE)
        if m:
            return thoughts, round(float(m.group(1)), 2)
    # last resort: a lone number on the final line
    last = text.splitlines()[-1] if text else ""
    m = re.search(r"(-?\d+(?:\.\d+)?)", last)
    return thoughts, (round(float(m.group(1)), 2) if m else None)


def cost_note(n_calls: int, models: Optional[List[str]] = None) -> str:
    models = sorted({m for m in (models or []) if m}) or [PAPER_MODEL]
    note = f"≈ {n_calls:,} model calls. Reasoning models bill their thinking tokens too."
    catalog = _catalog_url()  # endpoint-aware: don't print openrouter.ai links for a Nebius run
    if catalog == "https://openrouter.ai/models":
        links = " · ".join(f"https://openrouter.ai/{m.split(':', 1)[-1]}" for m in models)
        return f"{note} Rates: {links}"
    if catalog:
        return f"{note} Rates: {catalog}"
    return note
