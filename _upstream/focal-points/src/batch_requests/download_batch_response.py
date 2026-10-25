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
        safe_path_part,
    )
except ImportError:
    from batch_api_common import (
        gemini_api_key,
        object_to_dict,
        openai_api_key,
        resolve_job_record,
        safe_path_part,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a completed batch job response file."
    )
    parser.add_argument("--provider", default=None, help="Provider: gpt/openai or gemini.")
    parser.add_argument("--model", default=None, help="Model ID used for the batch.")
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
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/batch_requests_responses"),
        help="Root folder where response files will be saved.",
    )
    parser.add_argument(
        "--output-filename",
        default=None,
        help="Optional output filename. Defaults to <job-name>.jsonl.",
    )
    parser.add_argument("--api-key", default=None, help="Optional API key override.")
    return parser.parse_args()


def download_openai_response(
    job_name: str,
    output_dir: Path,
    output_filename: str,
    api_key: str | None,
) -> dict:
    client = OpenAI(api_key=openai_api_key(api_key))
    batch = client.batches.retrieve(job_name)
    batch_data = object_to_dict(batch)

    output_file_id = batch_data.get("output_file_id")
    if not output_file_id:
        raise RuntimeError(
            f"Batch {job_name} has no output_file_id yet. Current status: {batch_data.get('status')}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename
    content = client.files.content(output_file_id)
    output_path.write_bytes(content.content)

    error_path = None
    error_file_id = batch_data.get("error_file_id")
    if error_file_id:
        error_path = output_dir / f"{Path(output_filename).stem}.errors.jsonl"
        error_content = client.files.content(error_file_id)
        error_path.write_bytes(error_content.content)

    return {
        "batch": batch_data,
        "output_file": str(output_path),
        "error_file": str(error_path) if error_path else None,
    }


def download_gemini_response(
    job_name: str,
    output_dir: Path,
    output_filename: str,
    api_key: str | None,
) -> dict:
    key = gemini_api_key(api_key)
    client = genai.Client(api_key=key) if key else genai.Client()
    batch = client.batches.get(name=job_name)
    batch_data = object_to_dict(batch)

    state = batch_data.get("state")
    dest = batch_data.get("dest") or {}
    result_file_name = dest.get("fileName") or dest.get("file_name")
    if not result_file_name:
        raise RuntimeError(
            f"Batch {job_name} has no destination file yet. Current state: {state}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename
    content = client.files.download(file=result_file_name)
    output_path.write_bytes(content)

    return {
        "batch": batch_data,
        "output_file": str(output_path),
    }


def main() -> None:
    args = parse_args()
    record = resolve_job_record(
        job_name=args.job_name,
        job_file=args.job_file,
        batch_filename=args.batch_filename,
        provider=args.provider,
        model=args.model,
    )

    if not record.provider:
        raise ValueError("Could not determine provider; pass --provider")
    if not record.model:
        raise ValueError("Could not determine model; pass --model")

    output_dir = args.output_root / record.provider / safe_path_part(record.model)
    output_filename = args.output_filename or f"{safe_path_part(record.job_name)}.jsonl"

    if record.provider == "openai":
        result = download_openai_response(
            job_name=record.job_name,
            output_dir=output_dir,
            output_filename=output_filename,
            api_key=args.api_key,
        )
    elif record.provider == "gemini":
        result = download_gemini_response(
            job_name=record.job_name,
            output_dir=output_dir,
            output_filename=output_filename,
            api_key=args.api_key,
        )
    else:
        raise ValueError("provider must be openai/gpt or gemini")

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
