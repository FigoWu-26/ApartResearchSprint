"""
Responses API collusion tester for OpenAI reasoning models.
Uses reasoning models (o3, o4-mini, gpt-5.4, ...) with reasoning summaries exposed as 'thoughts'.

Calls follow the shared inference config (poseidon_compute): Nebius or OpenRouter,
OpenRouter as fallback for closed models (NEBIUS_API_KEY / OPENROUTER_API_KEY in compute.env).
OpenRouter exposes the same Responses API endpoint. Model ids are written in Nebius /
Hugging Face casing (openai/gpt-5.4; poseidon_compute translates to OpenRouter casing when it
routes there); a bare OpenAI name is prefixed automatically. Verified 2026-09-16 with
openai/gpt-5.4: `reasoning.summary` comes back through OpenRouter and is stored as 'thoughts'.

Reasoning summary options for --reasoning_summary:
  auto     - OpenAI decides length automatically (recommended)
  concise  - Short bullet-point summary of the reasoning
  detailed - Longer, more verbose summary

Note: temperature can be set with --temperature for supported models.
      If a specific model does not support sampling controls, the API may ignore
      or reject the value. Use --reasoning_effort to control compute.
"""
import os
import time
import json
import re
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from environment.oligopoly import OligopolyGame
import argparse

load_dotenv()

parser = argparse.ArgumentParser(description="OpenAI Responses API Collusion Tester")
parser.add_argument("--prefix", type=str, default="P0", help="Prefix for the game")
parser.add_argument("--prior_steps", type=int, default=1, help="Number of prior steps")
parser.add_argument("--model", type=str, default="openai/gpt-5.4", help="OpenRouter id of an OpenAI reasoning model (e.g. openai/gpt-5.4, openai/o3, openai/o4-mini)")
parser.add_argument("--reasoning_summary", type=str, default="auto", choices=["auto", "concise", "detailed"],
                    help="Reasoning summary verbosity: auto | concise | detailed")
parser.add_argument("--reasoning_effort", type=str, default="medium", choices=["low", "medium", "high"],
                    help="Reasoning effort (compute budget): low | medium | high")
parser.add_argument("--max_tokens", type=int, default=4096, help="Max output tokens per response")
parser.add_argument("--rounds", type=int, default=300, help="Number of game rounds to run")
parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature for supported models")
parser.add_argument("--max_retries", type=int, default=2, help="Number of extra retries after the initial failed generation")
parser.add_argument("--use_threadpool", action=argparse.BooleanOptionalAction, default=True, help="Call agents in parallel per step")
parser.add_argument("--seed", type=int, default=None, help="Run a single episode with this seed (0-9). Omit to run all 10.")
parser.add_argument("--overwrite", action="store_true", help="Overwrite any existing output file instead of resuming from the first failed or missing round")
args = parser.parse_args()

from poseidon_compute import llm_config
if not llm_config.api_key():
    raise ValueError("No API key found. Set NEBIUS_API_KEY (or OPENROUTER_API_KEY) in compute.env.")
client = llm_config.make_client()  # follows compute.env (OpenRouter / Nebius)

# OpenRouter ids are provider-qualified; accept the paper's bare OpenAI names too.
if "/" not in args.model:
    args.model = f"openai/{args.model}"


def is_successful_step(record):
    metrics = record.get("metrics", [])
    return bool(metrics) and all(m.get("success", False) for m in metrics)


def load_resumable_steps(filename):
    if not os.path.exists(filename):
        return []

    completed_steps = []
    with open(filename, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                break

            expected_round = len(completed_steps)
            if record.get("round") != expected_round or not is_successful_step(record):
                break

            completed_steps.append(record)

    return completed_steps


def extract_response_text(response):
    text = getattr(response, "output_text", "") or ""
    if text.strip():
        return text.strip()

    chunks = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) == "message":
            for content in getattr(item, "content", []) or []:
                content_text = getattr(content, "text", None)
                if content_text:
                    chunks.append(content_text)

    return "\n".join(chunks).strip()


def extract_price(response_text):
    patterns = [
        r"chosen\s*price\s*[:=]\s*\$?\s*(-?\d+(?:\.\d+)?)",
        r"price\s*[:=]\s*\$?\s*(-?\d+(?:\.\d+)?)",
        r"\$\s*(-?\d+(?:\.\d+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, response_text, flags=re.IGNORECASE)
        if match:
            return round(float(match.group(1)), 2)
    return None


def get_action(prompt, model=args.model):
    assert isinstance(prompt, list) and len(prompt) > 0
    assert prompt[0]["role"] == "system"
    assert prompt[1]["role"] == "user"

    start_time = time.time()
    metrics = {
        "request_start_time": start_time,
        "model": model,
        "temperature": args.temperature,
        "requested_reasoning_effort": args.reasoning_effort,
        "requested_reasoning_summary": args.reasoning_summary,
        "used_reduced_summary": False,
    }

    def create_response(reasoning_effort, reasoning_summary):
        request_kwargs = {
            "model": model,
            "input": prompt,
            "reasoning": {
                "effort": reasoning_effort,
                "summary": reasoning_summary,
            },
            "max_output_tokens": args.max_tokens,
        }
        if args.temperature is not None:
            request_kwargs["temperature"] = args.temperature
        return client.responses.create(**request_kwargs)

    try:
        final_reasoning_effort = args.reasoning_effort
        final_reasoning_summary = args.reasoning_summary
        response = create_response(final_reasoning_effort, final_reasoning_summary)

        response_status = getattr(response, "status", None)
        incomplete_details = getattr(response, "incomplete_details", None)
        incomplete_reason = getattr(incomplete_details, "reason", None) if incomplete_details else None

        if response_status == "incomplete" and incomplete_reason == "max_output_tokens":
            print("Response hit max_output_tokens before a final answer; retrying with reduced reasoning summary.")
            final_reasoning_effort = "low"
            final_reasoning_summary = "auto"
            metrics["used_reduced_summary"] = True
            response = create_response(final_reasoning_effort, final_reasoning_summary)
            response_status = getattr(response, "status", None)
            incomplete_details = getattr(response, "incomplete_details", None)
            incomplete_reason = getattr(incomplete_details, "reason", None) if incomplete_details else None

        thought = ""
        for item in response.output:
            if item.type == "reasoning" and item.summary:
                thought = item.summary[0].text
                break

        response_text = extract_response_text(response)
        price = extract_price(response_text)

        if price is None:
            print(f"[parse fail] raw response: {response_text!r}; status={response_status!r}; incomplete_reason={incomplete_reason!r}")
            metrics["parse_failure_response"] = response_text
            metrics["response_status"] = response_status
            metrics["incomplete_reason"] = incomplete_reason
            metrics["raw_response"] = response.model_dump(mode="json")
            raise ValueError(f"Failed to parse price from response: {response_text}")

        metrics.update({
            "request_end_time": time.time(),
            "latency_seconds": time.time() - start_time,
            "tokens_input": response.usage.input_tokens,
            "tokens_output": response.usage.output_tokens,
            "reasoning_tokens": response.usage.output_tokens_details.reasoning_tokens,
            "response_status": response_status,
            "incomplete_reason": incomplete_reason,
            "final_reasoning_effort": final_reasoning_effort,
            "final_reasoning_summary": final_reasoning_summary,
            "success": True,
        })

        return price, thought, response_text, metrics

    except Exception as e:
        metrics.update({
            "request_end_time": time.time(),
            "latency_seconds": time.time() - start_time,
            "error": str(e),
            "success": False,
        })
        print(f"Error in get_action: {e}")
        raise e


def retry_get_action(prompt, model, failures_file, max_retries=None):
    if max_retries is None:
        max_retries = args.max_retries
    max_retries = max(0, max_retries)

    attempt = 0
    last_error = None
    last_failed_response = None
    while attempt <= max_retries:
        try:
            return get_action(prompt, model)
        except Exception as e:
            last_error = e
            # Capture and immediately save parse failures
            if hasattr(e, 'args') and e.args and "Failed to parse price" in str(e.args[0]):
                last_failed_response = str(e.args[0]).replace("Failed to parse price from response: ", "")
                if failures_file is not None:
                    os.makedirs(os.path.dirname(failures_file), exist_ok=True)
                    with open(failures_file, "a") as ff:
                        ff.write(json.dumps({"retry": attempt, "response": last_failed_response}) + "\n")

            if attempt == max_retries:
                break

            attempt += 1
            print(f"Retry {attempt}/{max_retries}")

    print("Maximum retries reached. Giving up.")
    return (
        5.0,
        "Error: Failed after maximum retries",
        "Error: Failed to get response after multiple retries",
        {"success": False, "error": f"Failed after {max_retries} retries: {str(last_error)}",
         "parse_failure_response": last_failed_response},
    )


def get_actions(prompts, model=args.model, failures_file=None):
    actions, thoughts, responses, metrics_list = [], [], [], []

    if args.use_threadpool:
        with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
            future_results = [executor.submit(retry_get_action, p, model, failures_file) for p in prompts]
            for future in future_results:
                price, thought, response, metrics = future.result()
                actions.append(price)
                thoughts.append(thought)
                responses.append(response)
                metrics_list.append(metrics)
    else:
        for prompt in prompts:
            price, thought, response, metrics = retry_get_action(prompt, model, failures_file)
            actions.append(price)
            thoughts.append(thought)
            responses.append(response)
            metrics_list.append(metrics)

    return actions, thoughts, responses, metrics_list


def main():
    episodes = [args.seed] if args.seed is not None else list(range(10))
    safe_model = (args.model.replace("-", "_").replace(":", "_").replace(".", "_")).replace("/", "_")  # OpenRouter ids contain a slash
    safe_temp = str(args.temperature).replace("-", "m").replace(".", "p")
    failures_file = f"results/jsonl/openai_responses/parse_failures_{safe_model}_temp_{safe_temp}.jsonl"

    for episode in episodes:
        filename = (
            f"results/jsonl/openai_responses/"
            f"prior_{args.prior_steps}_prefix_{args.prefix}_model_{safe_model}_temp_{safe_temp}_seed_{episode}.jsonl"
        )
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        if args.overwrite and os.path.exists(filename):
            os.remove(filename)

        completed_steps = [] if args.overwrite else load_resumable_steps(filename)

        env = OligopolyGame(a0=0, ais=[2, 2], costs=[1, 1], mu=0.25, alpha=1.0, prefix=args.prefix)
        prompts = env.reset(prior_steps=args.prior_steps)

        for record in completed_steps[:args.rounds]:
            _, _, _, _, prompts = env.step(record["actions"])

        if os.path.exists(filename) and not args.overwrite:
            with open(filename, "w") as file:
                for record in completed_steps[:args.rounds]:
                    file.write(json.dumps(record) + "\n")

        if completed_steps:
            print(f"Resuming episode {episode} from round {len(completed_steps)}/{args.rounds}")

        episode_start = time.time()
        start_step = min(len(completed_steps), args.rounds)
        for step in range(start_step, args.rounds):
            step_start = time.time()
            actions, thoughts, responses, metrics_list = get_actions(prompts, failures_file=failures_file)

            prices, demands, profits, gains, prompts = env.step(actions)

            elapsed = time.time() - episode_start
            step_time = time.time() - step_start
            steps_done = step + 1
            eta = (elapsed / steps_done) * (args.rounds - steps_done)
            latencies = [f"{m.get('latency_seconds', 0):.1f}s" for m in metrics_list]
            print(f"episode= {episode}, round= {step}/{args.rounds}, elapsed= {elapsed:.0f}s, step= {step_time:.1f}s, ETA= {eta:.0f}s, prices= {actions}, demands= {demands}, profits= {profits}, gains= {gains}, latencies= {latencies}")

            for m in metrics_list:
                if "parse_failure_response" in m:
                    with open(failures_file, "a") as ff:
                        ff.write(json.dumps({"episode": episode, "step": step, "response": m["parse_failure_response"]}) + "\n")

            with open(filename, "a") as file:
                new_output = {
                    "round": step,
                    "actions": actions,
                    "thoughts": thoughts,
                    "respones": responses,  # kept original typo for consistency
                    "generation_metadata": [
                        {
                            "used_reduced_summary": m.get("used_reduced_summary", False),
                            "requested_reasoning_summary": m.get("requested_reasoning_summary"),
                            "final_reasoning_summary": m.get("final_reasoning_summary"),
                            "requested_reasoning_effort": m.get("requested_reasoning_effort"),
                            "final_reasoning_effort": m.get("final_reasoning_effort"),
                        }
                        for m in metrics_list
                    ],
                    "demands": demands,
                    "profits": profits,
                    "gains": gains,
                    "metrics": metrics_list,
                }
                file.write(json.dumps(new_output) + "\n")


if __name__ == "__main__":
    main()
