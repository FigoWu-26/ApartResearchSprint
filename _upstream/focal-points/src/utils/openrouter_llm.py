"""
LLM backend for the focal-point experiments (hackathon standard).

Every model call goes through the shared endpoint (**Nebius AI Studio** by default; see the repo-root
``compute.env`` / ``poseidon_compute``), so the experiments run on any laptop with no GPU. The paper's
default model (``meta-llama/Llama-3.3-70B-Instruct`` in ``experiments.sh`` / the Dockerfile) is the
default here too — model ids are used as-is, in Nebius / Hugging Face casing (the same ids the paper
used); nothing is rewritten or substituted.

Model specs:
  * a Hugging Face / Nebius model id, e.g. ``meta-llama/Llama-3.3-70B-Instruct``,
    ``Qwen/Qwen2.5-72B-Instruct``;
  * ``custom:<model>`` uses ``LLM_BASE_URL`` / ``LLM_API_KEY`` (any OpenAI-compatible
    server); ``ollama:<model>`` a free local Ollama. Not needed for the hackathon.

Sampling N responses per prompt is done with N separate calls (robust across providers and makes
resume trivial).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, Tuple


from .llm import APIBaseLLM


from poseidon_compute import endpoint_kind, make_client, resolve_model

DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"  # paper default (experiments.sh / Dockerfile)

# Fail fast instead of blocking the whole run on a single stuck request.
CLIENT_TIMEOUT = 120.0  # seconds per request
CLIENT_MAX_RETRIES = 2


@dataclass(frozen=True)
class Preset:
    base_url: str
    key_env: str
    needs_key: bool = True


PRESETS: Dict[str, Preset] = {
    "openrouter": Preset("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    # opt-in alternatives, not featured for the hackathon
    "ollama": Preset("http://localhost:11434/v1", "OLLAMA_API_KEY", needs_key=False),
    "custom": Preset("", "LLM_API_KEY", needs_key=False),
}


def parse_model_spec(spec: str) -> Tuple[str, str]:
    """'custom:foo' -> ('custom', 'foo'); a bare id -> (provider, id). Model ids are used as-is, in
    Nebius / Hugging Face casing (the same ids the paper used) — nothing is rewritten."""
    if ":" in spec and spec.split(":", 1)[0] in PRESETS:
        provider, model = spec.split(":", 1)
        return provider, model
    return os.getenv("LLM_PROVIDER", "openrouter"), spec


def resolve_model_id(spec: str) -> str:
    """The id used in output folder names (OpenRouter id, without a provider prefix)."""
    return parse_model_spec(spec)[1]


def cost_note(n_calls: int, model_spec: str) -> str:
    _, model = parse_model_spec(model_spec)
    # Show the model that will actually run (resolve_model applies the LLM_MODEL override, retired-pin
    # upgrades, and HF->OpenRouter casing) rather than the raw requested id, so the rate URL matches
    # the model that is actually billed.
    model = resolve_model(model)
    # Cite the rates page for the endpoint the calls will ACTUALLY use (Nebius, or OpenRouter
    # when that's the configured key), not a hardcoded provider — otherwise the banner misleads.
    kind = endpoint_kind()
    if kind == "openrouter":
        rates = f"https://openrouter.ai/{model}"
    elif kind == "nebius":
        rates = "Nebius AI Studio (https://studio.nebius.com)"
    else:  # custom endpoint (e.g. a Modal-hosted model)
        rates = "your configured endpoint"
    return f"≈ {n_calls:,} model calls. Per-token rates: {rates}"


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class OpenRouterLLM(APIBaseLLM):
    """Chat-completions client (OpenAI protocol) pointed at OpenRouter by default."""

    def __init__(
        self,
        model_id: str,
        reasoning: str | None = None,
        num_return_sequences: int = 1,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
        seed: int | None = None,
        top_p: float | None = None,
    ) -> None:
        self.provider, model = parse_model_spec(model_id)
        model = resolve_model(model)  # honor LLM_MODEL override; otherwise use the id as-is (no rewrite)
        preset = PRESETS[self.provider]
        self.base_url = os.getenv("LLM_BASE_URL") if self.provider == "custom" else preset.base_url
        if not self.base_url:
            raise ValueError("LLM_BASE_URL must be set for provider 'custom'")
        # The client authenticates with the key from the shared compute.env (via make_client() /
        # llm_config, Nebius or OpenRouter), which fails loudly if none is set. This field is only a
        # diagnostic hint for the preset; it is not what authenticates the client.
        self.api_key = os.getenv(preset.key_env)  # diagnostics only; real key comes from llm_config

        effort = reasoning.strip().lower() if reasoning else None
        if effort is not None and effort not in {"none", "minimal", "low", "medium", "high", "xhigh"}:
            raise ValueError(f"Invalid reasoning effort '{reasoning}' (use low / medium / high).")
        super().__init__(
            model_id=model,
            reasoning=effort,
            num_return_sequences=num_return_sequences,
            reasoning_display=effort or "Default",
        )
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.top_p = top_p
        self.is_local = "localhost" in self.base_url or "127.0.0.1" in self.base_url

        # Default provider ('openrouter' label) routes through the shared compute.env config
        # (Nebius or OpenRouter). The opt-in 'ollama' / 'custom' presets select a DIFFERENT endpoint,
        # so honor their resolved base_url/key instead of silently rerouting to the shared endpoint.
        # timeout / max_retries make a stuck request fail fast instead of blocking the whole run.
        if self.provider == "openrouter":
            self.client = make_client(timeout=CLIENT_TIMEOUT, max_retries=CLIENT_MAX_RETRIES)
        else:
            from openai import OpenAI

            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key or "EMPTY",
                timeout=CLIENT_TIMEOUT,
                max_retries=CLIENT_MAX_RETRIES,
            )
        print(f"[INFO] Initialized OpenRouterLLM ({self.provider}) with model_id: {self.model_id}")
        print(
            f"[INFO] Reasoning effort: {self.reasoning_display} | temperature: {temperature} | seed: {seed}"
        )
        print(f"[INFO] API sequences per prompt: {self.num_return_sequences}")

    def _generate_raw_text(self, prompt: str) -> str:
        kwargs: dict = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.max_new_tokens is not None:
            kwargs["max_tokens"] = self.max_new_tokens
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if self.reasoning and self.provider == "openrouter":
            if self.reasoning == "none":
                kwargs["extra_body"] = {"reasoning": {"enabled": False}}
            else:
                kwargs["extra_body"] = {"reasoning": {"effort": self.reasoning}}
        response = self.client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content or ""
        # reasoning models: drop the chain of thought, keep the answer
        return _THINK_RE.sub("", text).strip()

    def clear_cache(self) -> None:  # API model: nothing to free
        return None


def estimate_calls(prompt_plan, existing: dict, target_per_prompt: int) -> int:
    """Calls still needed to bring every prompt in the plan up to target_per_prompt."""
    n = 0
    for entry in prompt_plan:
        have = len(existing.get((entry["idx"], entry["prompt"]), []))
        n += max(0, target_per_prompt - have)
    return n
