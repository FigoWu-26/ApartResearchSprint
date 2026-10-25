import argparse
import json
from pathlib import Path

from google import genai
from openai import OpenAI

try:
    from .batch_api_common import (
        gemini_api_key,
        object_to_dict,
        openai_api_key,
        resolve_job_record,
    )
except ImportError:
    from batch_api_common import (
        gemini_api_key,
        object_to_dict,
        openai_api_key,
        resolve_job_record,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cancel an OpenAI or Gemini batch job."
    )
    parser.add_argument("--provider", default=None, help="Provider: gpt/openai or gemini.")
    parser.add_argument("--job-name", default=None, help="Batch job name/ID.")
    parser.add_argument(
        "--job-file",
        type=Path,
        default=None,
        help="Path to a job_name.txt file written by submit_batch_job.py.",
    )
    parser.add_argument(
        "--batch-filename",
        default=None,
        help="Batch filename to select from a multi-record job_name.txt file.",
    )
    parser.add_argument("--api-key", default=None, help="Optional API key override.")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually cancel the job. Without this flag the script only prints what it would cancel.",
    )
    return parser.parse_args()


def cancel_openai_job(job_name: str, api_key: str | None) -> dict:
    client = OpenAI(api_key=openai_api_key(api_key))
    return object_to_dict(client.batches.cancel(job_name))


def cancel_gemini_job(job_name: str, api_key: str | None) -> dict:
    key = gemini_api_key(api_key)
    client = genai.Client(api_key=key) if key else genai.Client()
    client.batches.cancel(name=job_name)
    return object_to_dict(client.batches.get(name=job_name))


def main() -> None:
    args = parse_args()
    record = resolve_job_record(
        job_name=args.job_name,
        job_file=args.job_file,
        batch_filename=args.batch_filename,
        provider=args.provider,
        model=None,
    )

    if not record.provider:
        raise ValueError("Could not determine provider; pass --provider")

    if not args.yes:
        print(
            "Dry run only. Re-run with --yes to cancel:\n"
            f"provider={record.provider}\n"
            f"job_name={record.job_name}\n"
            f"batch_file={record.batch_file}\n"
        )
        return

    if record.provider == "openai":
        result = cancel_openai_job(record.job_name, args.api_key)
    elif record.provider == "gemini":
        result = cancel_gemini_job(record.job_name, args.api_key)
    else:
        raise ValueError("provider must be openai/gpt or gemini")

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
