import argparse
import json
from pathlib import Path

from google import genai
from google.genai import types
from openai import OpenAI

try:
    from .batch_api_common import (
        append_or_replace_job_record,
        find_batch_file,
        gemini_api_key,
        normalize_provider,
        object_to_dict,
        openai_api_key,
        validate_openai_batch_model,
    )
except ImportError:
    from batch_api_common import (
        append_or_replace_job_record,
        find_batch_file,
        gemini_api_key,
        normalize_provider,
        object_to_dict,
        openai_api_key,
        validate_openai_batch_model,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit a JSONL batch request file to OpenAI or Gemini."
    )
    parser.add_argument(
        "json_filename",
        help="Batch JSONL filename or path. If only a filename is given, data/batch_requests is searched recursively.",
    )
    parser.add_argument(
        "--provider",
        required=True,
        help="API provider: gpt/openai or gemini.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Provider model ID, e.g. gpt-4o or gemini-3-flash-preview.",
    )
    parser.add_argument(
        "--batch-root",
        type=Path,
        default=Path("data/batch_requests"),
        help="Folder to search for batch request files.",
    )
    parser.add_argument(
        "--endpoint",
        default="/v1/chat/completions",
        help="OpenAI Batch endpoint. The generated OpenAI files currently use /v1/chat/completions.",
    )
    parser.add_argument(
        "--completion-window",
        default="24h",
        choices=["24h"],
        help="OpenAI Batch completion window.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional API key override. Otherwise OPENAI_API_KEY, GEMINI_API_KEY, or GOOGLE_API_KEY is used.",
    )
    parser.add_argument(
        "--skip-openai-model-validation",
        action="store_true",
        help="Do not verify that every OpenAI request body uses --model before submitting.",
    )
    return parser.parse_args()


def submit_openai_batch(
    batch_file: Path,
    model: str,
    endpoint: str,
    completion_window: str,
    api_key: str | None,
) -> tuple[str, dict]:
    client = OpenAI(api_key=openai_api_key(api_key))

    with open(batch_file, "rb") as f:
        uploaded_file = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=uploaded_file.id,
        endpoint=endpoint,
        completion_window=completion_window,
        metadata={
            "source_file": batch_file.name,
            "model": model,
        },
    )

    return batch.id, {
        "uploaded_file": object_to_dict(uploaded_file),
        "batch": object_to_dict(batch),
    }


def submit_gemini_batch(
    batch_file: Path,
    model: str,
    api_key: str | None,
) -> tuple[str, dict]:
    key = gemini_api_key(api_key)
    client = genai.Client(api_key=key) if key else genai.Client()

    uploaded_file = client.files.upload(
        file=batch_file,
        config=types.UploadFileConfig(
            display_name=batch_file.stem,
            mime_type="jsonl",
        ),
    )
    batch = client.batches.create(
        model=model,
        src=uploaded_file.name,
        config=types.CreateBatchJobConfig(display_name=batch_file.stem),
    )

    return batch.name, {
        "uploaded_file": object_to_dict(uploaded_file),
        "batch": object_to_dict(batch),
    }


def main() -> None:
    args = parse_args()
    provider = normalize_provider(args.provider)
    batch_file = find_batch_file(args.json_filename, args.batch_root)

    if provider == "openai" and not args.skip_openai_model_validation:
        validate_openai_batch_model(batch_file, args.model)

    if provider == "openai":
        job_name, details = submit_openai_batch(
            batch_file=batch_file,
            model=args.model,
            endpoint=args.endpoint,
            completion_window=args.completion_window,
            api_key=args.api_key,
        )
    else:
        job_name, details = submit_gemini_batch(
            batch_file=batch_file,
            model=args.model,
            api_key=args.api_key,
        )

    job_file = batch_file.parent / "job_name.txt"
    append_or_replace_job_record(
        job_file=job_file,
        batch_file=batch_file,
        provider=provider,
        model=args.model,
        job_name=job_name,
    )

    print(json.dumps(details, indent=2, ensure_ascii=False))
    print(f"job_name: {job_name}")
    print(f"saved_job_file: {job_file}")


if __name__ == "__main__":
    main()
