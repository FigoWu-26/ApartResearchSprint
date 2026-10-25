"""OpenAI-compatible vLLM endpoint on Modal for the open-source agent model.

Deploy once (persistent; scales to zero after `SCALEDOWN_MINUTES` idle):

    modal deploy modal_infra/serve_vllm.py

then put the printed URL (+ "/v1") in the repo-root compute.env as MODAL_LLM_BASE_URL, and the key stored in
the `collusion-vllm` Modal secret as MODAL_LLM_API_KEY. Agents select this endpoint with model ids prefixed
`modal:` (e.g. the `qwen3-32b` alias in cda_types.MODEL_ALIASES).

Unlike util/poseidon_compute/modal_serve.py (vLLM 0.6.6, one request per container), this pins a vLLM that
supports Qwen3 and lets one replica batch many concurrent agent calls.
"""

import json
import os
import subprocess

import modal

MODEL_NAME = os.environ.get("SERVE_MODEL", "Qwen/Qwen3-32B-FP8")
MODEL_REVISION = os.environ.get("SERVE_MODEL_REVISION", "aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df")
GPU = os.environ.get("SERVE_GPU", "H100")
MAX_INPUTS = int(os.environ.get("SERVE_MAX_INPUTS", "32"))  # concurrent requests per replica
MAX_CONTAINERS = int(os.environ.get("SERVE_MAX_CONTAINERS", "4"))  # caps spend
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


@app.function(
    image=image,
    gpu=GPU,
    scaledown_window=SCALEDOWN_MINUTES * MINUTES,
    timeout=24 * 60 * MINUTES,
    max_containers=MAX_CONTAINERS,
    volumes={"/root/.cache/huggingface": hf_cache_vol, "/root/.cache/vllm": vllm_cache_vol},
    secrets=[modal.Secret.from_name("collusion-vllm")],  # VLLM_API_KEY
)
@modal.concurrent(max_inputs=MAX_INPUTS)
@modal.web_server(port=VLLM_PORT, startup_timeout=30 * MINUTES)
def serve():
    cmd = [
        "vllm", "serve", MODEL_NAME,
        "--revision", MODEL_REVISION,
        "--served-model-name", MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--api-key", os.environ["VLLM_API_KEY"],
        "--max-model-len", "16384",
        "--max-num-seqs", str(MAX_INPUTS),
        "--max-cudagraph-capture-size", str(MAX_INPUTS),
        "--reasoning-parser", "qwen3",  # separates <think> text if a request enables thinking
        "--generation-config", "vllm",  # sampling comes from the request, not the model's generation_config.json
    ]
    print(json.dumps([c for c in cmd if c != os.environ["VLLM_API_KEY"]]))
    subprocess.Popen(cmd)
