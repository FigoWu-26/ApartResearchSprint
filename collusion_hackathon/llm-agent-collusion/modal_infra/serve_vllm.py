"""OpenAI-compatible vLLM endpoints on Modal for the live agent models.

Deploy once (persistent; each server scales to zero after `SCALEDOWN_MINUTES` idle):

    modal deploy modal_infra/serve_vllm.py

Then put the printed URLs (+ "/v1") in the repo-root compute.env:

    MODAL_LLM_BASE_URL    Qwen3 32B      function `serve`
    MODAL_KIMI_BASE_URL   Kimi VL A3B    function `serve_kimi`
    MODAL_GEMMA_BASE_URL  Gemma 3 27B    function `serve_gemma`

and the key stored in the `collusion-vllm` Modal secret as MODAL_LLM_API_KEY.
Agents select a server with the aliases in modal_infra/models.py (`qwen3-32b`, `kimi`, `gemma`),
which the client sends as model ids prefixed `modal:`.

Gemma 3 weights are license-gated. The Gemma server reads HF_TOKEN from the `huggingface-secret`
Modal secret; the token's Hugging Face account must have accepted the Gemma license.

Unlike util/poseidon_compute/modal_serve.py (vLLM 0.6.6, one request per container), this pins a vLLM that
supports these models and lets one replica batch many concurrent agent calls.
"""

import json
import os
import subprocess

import modal

QWEN_REPO = "Qwen/Qwen3-32B-FP8"
QWEN_REVISION = "aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df"
KIMI_REPO = "moonshotai/Kimi-VL-A3B-Instruct"
KIMI_REVISION = "398eede0903cd983a2bfa0cc634e9ac1d843f375"
GEMMA_REPO = "google/gemma-3-27b-it"
GEMMA_REVISION = "005ad3404e59d6023443cb575daa05336842228a"

GPU = os.environ.get("SERVE_GPU", "H100")
MAX_INPUTS = int(os.environ.get("SERVE_MAX_INPUTS", "32"))  # concurrent requests per Qwen replica
MAX_CONTAINERS = int(os.environ.get("SERVE_MAX_CONTAINERS", "4"))  # caps Qwen spend
SCALEDOWN_MINUTES = 15
MINUTES = 60
VLLM_PORT = 8000

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.18.1", "huggingface_hub[hf_transfer]==0.36.0")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("collusion-vllm")
api_secret = modal.Secret.from_name("collusion-vllm")  # VLLM_API_KEY
hf_secret = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])


def _launch(repo: str, revision: str, max_seqs: int, extra: list[str]) -> None:
    cmd = [
        "vllm", "serve", repo,
        "--revision", revision,
        "--served-model-name", repo,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--api-key", os.environ["VLLM_API_KEY"],
        "--max-model-len", "16384",
        "--max-num-seqs", str(max_seqs),
        "--max-cudagraph-capture-size", str(max_seqs),
        "--generation-config", "vllm",  # sampling comes from the request, not generation_config.json
        *extra,
    ]
    print(json.dumps([c for c in cmd if c != os.environ["VLLM_API_KEY"]]))
    subprocess.Popen(cmd)


@app.function(
    image=image,
    gpu=GPU,
    scaledown_window=SCALEDOWN_MINUTES * MINUTES,
    timeout=24 * 60 * MINUTES,
    max_containers=MAX_CONTAINERS,
    volumes={"/root/.cache/huggingface": hf_cache_vol, "/root/.cache/vllm": vllm_cache_vol},
    secrets=[api_secret],
)
@modal.concurrent(max_inputs=MAX_INPUTS)
@modal.web_server(port=VLLM_PORT, startup_timeout=30 * MINUTES)
def serve():
    _launch(QWEN_REPO, QWEN_REVISION, MAX_INPUTS, [
        "--reasoning-parser", "qwen3",  # separates <think> text if a request enables thinking
    ])


@app.function(
    image=image,
    gpu=GPU,
    scaledown_window=SCALEDOWN_MINUTES * MINUTES,
    timeout=24 * 60 * MINUTES,
    max_containers=2,
    volumes={"/root/.cache/huggingface": hf_cache_vol, "/root/.cache/vllm": vllm_cache_vol},
    secrets=[api_secret],
)
@modal.concurrent(max_inputs=16)
@modal.web_server(port=VLLM_PORT, startup_timeout=30 * MINUTES)
def serve_kimi():
    # Text-only agent calls. The vision tower stays in the checkpoint; image prompts are not accepted.
    _launch(KIMI_REPO, KIMI_REVISION, 16, [
        "--trust-remote-code",
        "--limit-mm-per-prompt", json.dumps({"image": 0}),
    ])


@app.function(
    image=image,
    gpu=GPU,
    scaledown_window=SCALEDOWN_MINUTES * MINUTES,
    timeout=24 * 60 * MINUTES,
    max_containers=2,
    volumes={"/root/.cache/huggingface": hf_cache_vol, "/root/.cache/vllm": vllm_cache_vol},
    secrets=[api_secret, hf_secret],
)
@modal.concurrent(max_inputs=8)
@modal.web_server(port=VLLM_PORT, startup_timeout=30 * MINUTES)
def serve_gemma():
    # 27B bf16 weights on one H100. Fewer sequences leave room for the 16k context the agents use.
    _launch(GEMMA_REPO, GEMMA_REVISION, 8, [
        "--gpu-memory-utilization", "0.95",
        "--limit-mm-per-prompt", json.dumps({"image": 0}),
    ])
