from abc import ABC, abstractmethod
import gc
import os
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_fixed

# Hackathon note: torch / transformers (local GPU inference), google-genai and the
# OpenAI client are imported lazily inside the classes that need them, so the core
# install stays CPU-only. The default backend is OpenRouterLLM (see openrouter_llm.py).


class LLM(ABC):
    def __init__(self, model_id: str, is_api_model: bool) -> None:
        self.model_id = model_id
        self.is_api_model = is_api_model

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a single response for the provided prompt."""
        raise NotImplementedError

    @abstractmethod
    def generate_batch(
        self, prompts: list[str], num_return_sequences: int | None = None
    ) -> list[list[str]]:
        """Generate responses for a list of prompts.

        Args:
            prompts: Prompts to pass to the model.
            num_return_sequences: Optional override for the number of responses per
                prompt. If None, fall back to the model's configured default.
        """
        raise NotImplementedError


class LocalLLM(LLM):
    MODELS = {
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "meta-llama/Llama-3.1-405B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
        "meta-llama/Llama-3.1-70B-Instruct",
        "meta-llama/Meta-Llama-3-70B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "Qwen/Qwen2-72B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct-1M",
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct-1M",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2-7B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2-1.5B-Instruct",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2-0.5B-Instruct",
        "google/gemma-3-1b-it",
        "google/gemma-3-4b-it",
        "google/gemma-3-12b-it",
        "google/gemma-3-27b-it",
        "microsoft/Phi-4-mini-instruct",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
    }

    def __init__(
        self,
        model_id: str,
        top_p: float | None = None,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
        num_return_sequences: int | None = None,
        quantization: str | None = None,  # None, "8bit", or "4bit"
        reasoning: str | None = None,
    ):
        super().__init__(model_id=model_id, is_api_model=False)
        try:
            import torch
            from transformers import (  # noqa: F401
                AutoModelForCausalLM,
                AutoTokenizer,
                pipeline,
                BitsAndBytesConfig,
            )
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "Local inference needs the optional GPU extra: `uv sync --extra gpu`. "
                "For the hackathon use the default OpenRouter backend instead."
            ) from e
        self._torch = torch
        globals().update(
            torch=torch,
            AutoModelForCausalLM=AutoModelForCausalLM,
            AutoTokenizer=AutoTokenizer,
            pipeline=pipeline,
            BitsAndBytesConfig=BitsAndBytesConfig,
        )
        self.top_p = top_p
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.num_return_sequences = num_return_sequences
        self.quantization = quantization
        self.reasoning = None

        if reasoning:
            local_reasoning = reasoning.strip().lower()
            if local_reasoning in {"low", "medium", "high"}:
                self.reasoning = local_reasoning
                print(f"[INFO] Local model reasoning level: {self.reasoning}")
            else:
                self.reasoning = None
                print(
                    "[WARNING] Invalid reasoning level provided; ignoring reasoning parameter and using default."
                )

        # Load the tokenizer (same for all cases)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=os.getenv("HF_TOKEN"),
        )

        # Build a BitsAndBytesConfig only if quantization is requested
        bnb_config = None

        openai_models = {"openai/gpt-oss-120b", "openai/gpt-oss-20b"}
        if self.max_new_tokens is None and self.model_id in openai_models:
            # Default longer generations for OSS OpenAI models
            self.max_new_tokens = 4096

        if quantization == "8bit" and self.model_id not in openai_models:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        elif quantization == "4bit" and self.model_id not in openai_models:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        # Prepare kwargs for from_pretrained()
        model_kwargs: dict = {
            "token": os.getenv("HF_TOKEN"),
            "trust_remote_code": True,
            "device_map": "auto",
        }

        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config
        else:
            use_cuda = torch.cuda.is_available()
            model_kwargs["torch_dtype"] = torch.bfloat16 if use_cuda else torch.float32

        # Load the model once
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_kwargs,
        )

        self._log_device_allocation()

        # Build the generation kwargs dictionary
        generation_kwargs: dict = {
            "do_sample": True,
            "return_full_text": False,
        }
        if self.top_p is not None:
            generation_kwargs["top_p"] = self.top_p
        if self.temperature is not None:
            generation_kwargs["temperature"] = self.temperature
        if self.max_new_tokens is not None:
            generation_kwargs["max_new_tokens"] = self.max_new_tokens
        if self.num_return_sequences is not None:
            generation_kwargs["num_return_sequences"] = self.num_return_sequences

        # Create a text-generation pipeline that accepts a batch of prompts
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            disable_compile=(True if self.model_id == "google/gemma-3-4b-it" else None),
            # this fixes this bug with Gemma: https://github.com/huggingface/transformers/issues/38333
            **generation_kwargs,
        )

        system_prompt = "You are a helpful assistant."
        if self.model_id in openai_models and self.reasoning:
            system_prompt = f"{system_prompt}\nReasoning: {self.reasoning}"

        self.chat_template = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": None},
        ]

        # Print system prompt for verification
        print(f"[INFO] System prompt set to: \n{system_prompt}")

    def _log_device_allocation(self) -> None:
        if not self._torch.cuda.is_available():
            print("[INFO] CUDA not available; skipping device allocation check.")
            return

        device_map = getattr(self.model, "hf_device_map", None)
        if device_map:
            print("[INFO] Model device map:")
            for module, device in device_map.items():
                print(f"  - {module}: {device}")
        else:
            print(
                "[INFO] Model did not expose a device map; assuming single-device placement."
            )

        for idx in range(self._torch.cuda.device_count()):
            allocated_gib = self._torch.cuda.memory_allocated(idx) / (1024**3)
            reserved_gib = self._torch.cuda.memory_reserved(idx) / (1024**3)
            print(
                f"[INFO] CUDA:{idx} memory allocated: {allocated_gib:.2f} GiB "
                f"(reserved {reserved_gib:.2f} GiB)"
            )

    def clear_cache(self):
        # free VRAM / system RAM
        torch = self._torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        del self.generator, self.model
        gc.collect()

    def generate_batch(
        self, prompts: list[str], num_return_sequences: int | None = None
    ) -> list[list[str]]:
        formatted_chats = []
        for prompt in prompts:
            chat = [
                {"role": "system", "content": self.chat_template[0]["content"]},
                {"role": "user", "content": prompt},
            ]
            formatted_chats.append(chat)

        generator_kwargs: dict = {}
        if num_return_sequences is not None:
            generator_kwargs["num_return_sequences"] = num_return_sequences

        results = self.generator(formatted_chats, **generator_kwargs)
        all_responses: list[list[str]] = []

        for out in results:
            if isinstance(out, list):
                texts = [
                    "".join(
                        c
                        for c in entry["generated_text"].strip().lower()
                        if c.isalnum() or c.isspace() or c in "<>/"
                    )
                    for entry in out
                ]
            else:
                texts = [
                    "".join(
                        c
                        for c in out["generated_text"].strip().lower()
                        if c.isalnum() or c.isspace() or c in "<>/"
                    )
                ]
            all_responses.append(texts)

        return all_responses

    def generate(self, prompt: str) -> str:
        return self.generate_batch([prompt])[0][0]


class APIBaseLLM(LLM):
    HIGH_DEMAND_CODE = 503
    API_MAX_RETRY_ATTEMPTS = 240
    API_RETRY_SLEEP_SECONDS = 30

    @staticmethod
    def _get_status_code(error: BaseException) -> int | None:
        status_code = getattr(error, "status_code", None)
        if status_code is None:
            response = getattr(error, "response", None)
            status_code = getattr(response, "status_code", None)
        if status_code is None:
            return None
        try:
            return int(status_code)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _is_high_demand_api_error(error: BaseException) -> bool:
        if APIBaseLLM._get_status_code(error) == APIBaseLLM.HIGH_DEMAND_CODE:
            return True

        error_message = str(error).lower()
        return any(marker in error_message for marker in ("high demand",))

    @staticmethod
    def _log_api_retry(retry_state) -> None:
        error = retry_state.outcome.exception()
        status_code = APIBaseLLM._get_status_code(error)
        status_text = f", status={status_code}" if status_code is not None else ""
        error_text = str(error).strip() or "<no error message>"
        print(f"[WARNING] {type(error).__name__}{status_text}: {error_text}", flush=True)

    def __init__(
        self,
        model_id: str,
        reasoning: object | None = None,
        num_return_sequences: int = 1,
        reasoning_display: str = "Default",
    ) -> None:
        super().__init__(model_id=model_id, is_api_model=True)
        self.reasoning = reasoning
        self.reasoning_display = reasoning_display
        self.num_return_sequences = num_return_sequences

    def _normalize_output(self, text: str) -> str:
        return "".join(
            c for c in text.strip().lower() if c.isalnum() or c.isspace() or c in "<>/"
        )

    @abstractmethod
    def _generate_raw_text(self, prompt: str) -> str:
        raise NotImplementedError

    @retry(
        stop=stop_after_attempt(API_MAX_RETRY_ATTEMPTS),
        wait=wait_fixed(API_RETRY_SLEEP_SECONDS),
        retry=retry_if_exception(_is_high_demand_api_error),
        before_sleep=_log_api_retry,
        reraise=True,
    )
    def _generate_raw_text_with_retry(self, prompt: str) -> str:
        return self._generate_raw_text(prompt)

    def generate_batch(
        self, prompts: list[str], num_return_sequences: int | None = None
    ) -> list[list[str]]:
        all_outputs: list[list[str]] = []
        for prompt in prompts:
            prompt_outputs: list[str] = []
            sequences_to_generate = (
                num_return_sequences
                if num_return_sequences is not None
                else self.num_return_sequences
            )
            for _ in range(sequences_to_generate):
                prompt_outputs.append(
                    self._normalize_output(self._generate_raw_text_with_retry(prompt))
                )
            all_outputs.append(prompt_outputs)
        return all_outputs

    def generate(self, prompt: str) -> str:
        return self.generate_batch([prompt])[0][0]


class OpenAIAPILLM(APIBaseLLM):
    MODELS = {"gpt-4o", "gpt-5", "gpt-5.1", "gpt-5.4"}
    REASONING_LEVELS_BY_MODEL = {
        "gpt-5": {"minimal", "low", "medium", "high"},
        "gpt-5.1": {"none", "low", "medium", "high"},
        "gpt-5.4": {"none", "low", "medium", "high", "xhigh"},
    }

    def __init__(
        self,
        model_id: str,
        reasoning: str | None = None,
        num_return_sequences: int = 1,
    ) -> None:
        reasoning_payload = None
        reasoning_display = "Default"
        if reasoning is not None:
            if model_id not in self.REASONING_LEVELS_BY_MODEL:
                raise ValueError(f"{model_id} does not support GPT reasoning levels.")

            gpt_reasoning = reasoning.strip().lower()
            supported_levels = self.REASONING_LEVELS_BY_MODEL[model_id]
            if gpt_reasoning not in supported_levels:
                raise ValueError(
                    f"Invalid GPT reasoning level '{reasoning}' for "
                    f"{model_id}. Supported levels: {', '.join(sorted(supported_levels))}."
                )
            reasoning_payload = {"effort": gpt_reasoning}
            reasoning_display = gpt_reasoning

        super().__init__(
            model_id=model_id,
            reasoning=reasoning_payload,
            num_return_sequences=num_return_sequences,
            reasoning_display=reasoning_display,
        )
        from openai import OpenAI

        _openai_key = os.getenv("OPENAI_API_KEY")
        if not _openai_key:
            raise RuntimeError(
                "OpenAIAPILLM talks to OpenAI directly (GPT reasoning models via the Responses API), "
                "which needs OPENAI_API_KEY. This path is OpenAI-only; the hackathon default is Nebius "
                "(see COMPUTE.md). Either set OPENAI_API_KEY, or use an open model through the shared "
                "llm_config client instead of this class."
            )
        self.client = OpenAI(api_key=_openai_key)

        print(f"[INFO] Initialized OpenAIAPILLM with model_id: {self.model_id}")
        print(f"[INFO] GPT reasoning level: {self.reasoning_display}")
        print(f"[INFO] API sequences per prompt: {self.num_return_sequences}")

    def _extract_output_text(self, response) -> str:
        text_chunks: list[str] = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) == "message":
                for content in getattr(item, "content", []):
                    if getattr(content, "type", None) == "output_text":
                        text_chunks.append(content.text)
            elif getattr(item, "type", None) == "output_text":
                text_chunks.append(item.text)
        if not text_chunks and getattr(response, "output_text", None):
            text_chunks.append(response.output_text)
        return "".join(text_chunks).strip()

    def _generate_raw_text(self, prompt: str) -> str:
        request_kwargs = {
            "model": self.model_id,
            "input": prompt,
        }
        if self.reasoning is not None:
            request_kwargs["reasoning"] = self.reasoning

        response = self.client.responses.create(**request_kwargs)
        return self._extract_output_text(response)


class GeminiAPILLM(APIBaseLLM):
    MODELS = {
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite-preview",
        "gemini-3.1-pro-preview",
    }
    REASONING_LEVELS = {"none", "minimal", "low", "medium", "high"}

    def __init__(
        self,
        model_id: str,
        reasoning: str | None = None,
        num_return_sequences: int = 1,
    ) -> None:
        gemini_reasoning = reasoning.strip().lower() if reasoning is not None else None
        if gemini_reasoning not in {None, *self.REASONING_LEVELS}:
            raise ValueError(
                f"Invalid Gemini thinking level '{reasoning}'. Supported levels: "
                f"{', '.join(sorted(self.REASONING_LEVELS))}."
            )
        reasoning_display = "Default" if gemini_reasoning is None else gemini_reasoning

        super().__init__(
            model_id=model_id,
            reasoning=gemini_reasoning,
            num_return_sequences=num_return_sequences,
            reasoning_display=reasoning_display,
        )
        from google import genai

        self.client = genai.Client()

        print(f"[INFO] Initialized GeminiAPILLM with model_id: {self.model_id}")
        print(f"[INFO] Gemini thinking level: {self.reasoning_display}")
        print(f"[INFO] API sequences per prompt: {self.num_return_sequences}")

    def _generate_raw_text(self, prompt: str) -> str:
        from google.genai import types

        request_kwargs = {
            "model": self.model_id,
            "contents": prompt,
        }
        if self.reasoning == "none":
            request_kwargs["config"] = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        elif self.reasoning:
            request_kwargs["config"] = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level=self.reasoning)
            )

        response = self.client.models.generate_content(**request_kwargs)
        return response.text or ""


def load_model(
    model_id: str,
    top_p: float | None = None,
    temperature: float | None = None,
    max_new_tokens: int | None = None,
    num_return_sequences: int = 1,
    quantization: str | None = None,
    reasoning: str | None = None,
    backend: str = "openrouter",
    seed: int | None = None,
) -> LLM:
    """
    Load an LLM implementation.

    backend="openrouter" (default, hackathon standard): every model id - including the
    paper's HuggingFace ids - is served through the shared inference endpoint, no GPU needed.
    backend="original": the paper's dispatch (OpenAI Responses API / Gemini / local
    transformers) - needs the `providers` and/or `gpu` extras and the matching keys.
    """
    if backend == "openrouter":
        from .openrouter_llm import OpenRouterLLM

        return OpenRouterLLM(
            model_id=model_id,
            reasoning=reasoning,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            seed=seed,
            top_p=top_p,
        )

    if model_id in OpenAIAPILLM.MODELS:
        return OpenAIAPILLM(
            model_id=model_id,
            reasoning=reasoning,
            num_return_sequences=num_return_sequences,
        )
    elif model_id in GeminiAPILLM.MODELS:
        return GeminiAPILLM(
            model_id=model_id,
            reasoning=reasoning,
            num_return_sequences=num_return_sequences,
        )
    elif model_id in LocalLLM.MODELS:
        return LocalLLM(
            model_id=model_id,
            top_p=top_p,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences,
            quantization=quantization,
            reasoning=reasoning,
        )

    raise ValueError(f"Model ID '{model_id}' is not recognized as a valid model.")
