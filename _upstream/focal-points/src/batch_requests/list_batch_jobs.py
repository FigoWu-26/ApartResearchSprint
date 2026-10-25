import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from openai import OpenAI

try:
    from .batch_api_common import (
        gemini_api_key,
        normalize_provider,
        object_to_dict,
        openai_api_key,
        read_job_records,
    )
except ImportError:
    from batch_api_common import (
        gemini_api_key,
        normalize_provider,
        object_to_dict,
        openai_api_key,
        read_job_records,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List OpenAI/Gemini batch jobs and local job_name.txt records."
    )
    parser.add_argument(
        "--provider",
        default="all",
        help="Provider to query: all, gpt/openai, or gemini.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum jobs to request per provider.",
    )
    parser.add_argument(
        "--batch-root",
        type=Path,
        default=Path("data/batch_requests"),
        help="Folder containing local job_name.txt files.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional API key override. Only valid when querying a single provider.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON instead of a compact table.",
    )
    return parser.parse_args()


def selected_providers(provider: str) -> list[str]:
    if provider.strip().lower() == "all":
        return ["openai", "gemini"]
    return [normalize_provider(provider)]


def load_local_records(batch_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job_file in sorted(batch_root.rglob("job_name.txt")):
        for record in read_job_records(job_file):
            rows.append(
                {
                    "job_file": str(job_file),
                    "batch_file": record.batch_file,
                    "provider": record.provider,
                    "model": record.model,
                    "job_name": record.job_name,
                }
            )
    return rows


def list_openai_jobs(limit: int, api_key: str | None) -> list[dict[str, Any]]:
    key = openai_api_key(api_key)
    if not key:
        return []

    client = OpenAI(api_key=key)
    jobs: list[dict[str, Any]] = []
    for batch in client.batches.list(limit=limit):
        jobs.append(object_to_dict(batch))
        if len(jobs) >= limit:
            break
    return jobs


def list_gemini_jobs(limit: int, api_key: str | None) -> list[dict[str, Any]]:
    key = gemini_api_key(api_key)
    if not key:
        return []

    client = genai.Client(api_key=key)
    pager = client.batches.list(config=types.ListBatchJobsConfig(page_size=limit))
    jobs: list[dict[str, Any]] = []
    for batch in pager:
        jobs.append(object_to_dict(batch))
        if len(jobs) >= limit:
            break
    return jobs


def job_state(provider: str, job: dict[str, Any]) -> str:
    if provider == "openai":
        return str(job.get("status") or "unknown")
    return str(job.get("state") or "unknown")


def job_name(provider: str, job: dict[str, Any]) -> str:
    return str(job.get("id") if provider == "openai" else job.get("name"))


def job_model(provider: str, job: dict[str, Any]) -> str:
    if provider == "openai":
        metadata = job.get("metadata") or {}
        return str(metadata.get("model") or "unknown")
    return str(job.get("model") or "unknown")


def job_source(provider: str, job: dict[str, Any]) -> str:
    if provider == "openai":
        metadata = job.get("metadata") or {}
        return str(metadata.get("source_file") or job.get("input_file_id") or "")
    src = job.get("src") or {}
    return str(src.get("fileName") or src.get("file_name") or src or "")


def print_table(provider_jobs: dict[str, list[dict[str, Any]]], local_records: list[dict[str, Any]]) -> None:
    for provider, jobs in provider_jobs.items():
        counts = Counter(job_state(provider, job) for job in jobs)
        print(f"{provider}: {len(jobs)} jobs returned")
        if counts:
            print("  states: " + ", ".join(f"{state}={count}" for state, count in sorted(counts.items())))
        for job in jobs:
            print(
                "  "
                f"{job_name(provider, job)} | "
                f"{job_state(provider, job)} | "
                f"{job_model(provider, job)} | "
                f"{job_source(provider, job)}"
            )
        print()

    print(f"local job_name.txt records: {len(local_records)}")
    for record in local_records:
        print(
            "  "
            f"{record['provider']} | "
            f"{record['model']} | "
            f"{record['batch_file']} | "
            f"{record['job_name']} | "
            f"{record['job_file']}"
        )


def main() -> None:
    args = parse_args()
    providers = selected_providers(args.provider)
    if args.api_key and len(providers) != 1:
        raise ValueError("--api-key can only be used when --provider is not all")

    provider_jobs: dict[str, list[dict[str, Any]]] = {}
    for provider in providers:
        if provider == "openai":
            provider_jobs[provider] = list_openai_jobs(args.limit, args.api_key)
        elif provider == "gemini":
            provider_jobs[provider] = list_gemini_jobs(args.limit, args.api_key)
        else:
            raise ValueError("provider must be all, openai/gpt, or gemini")

    local_records = load_local_records(args.batch_root)

    if args.json:
        print(
            json.dumps(
                {
                    "provider_jobs": provider_jobs,
                    "local_records": local_records,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        print_table(provider_jobs, local_records)


if __name__ == "__main__":
    main()
