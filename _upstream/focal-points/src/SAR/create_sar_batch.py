import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from openai import OpenAI

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.batch_requests.batch_api_common import (
    object_to_dict,
    openai_api_key,
    safe_path_part,
)

DEFAULT_DATASET_CSV = Path("data/SAR_prompts/InitialConditions.csv")
DEFAULT_PROMPTS = Path("data/SAR_prompts/sar_prompts.json")
DEFAULT_MAP_DIR = Path("data/SAR_maps/v1/usgs_terrain")
DEFAULT_OUTPUT_DIR = Path("data/SAR_prompts/batch_requests")
DEFAULT_JOB_LOG = Path("data/SAR_prompts/job_ids.jsonl")
RESPONSES_ENDPOINT = "/v1/responses"


@dataclass(frozen=True)
class InitialCondition:
    incident_index: int
    ipp_lat: float
    ipp_lon: float
    find_lat: float
    find_lon: float


@dataclass(frozen=True)
class PromptVariant:
    name: str
    version: str
    text: str


@dataclass(frozen=True)
class RunContext:
    timestamp: datetime
    row_spec: str
    model_slug: str
    run_slug: str
    batch_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class SubmittedBatch:
    uploaded_batch_file: Any
    batch: Any
    request_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create and submit an OpenAI Batch API job for SAR map prompts. "
            "The script uploads each map once, reuses its file_id across repeats, "
            "and appends the submitted batch id to a local job log."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        help="OpenAI model id to use in each /v1/responses request, e.g. gpt-5.5.",
    )
    parser.add_argument(
        "--dataset-csv",
        type=Path,
        default=DEFAULT_DATASET_CSV,
        help="CSV with incident_index, IPP_lat/lon, and find_lat/lon.",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=DEFAULT_PROMPTS,
        help="Prompt JSON with top-level prompt names and version keys.",
    )
    parser.add_argument(
        "--map-dir",
        type=Path,
        default=DEFAULT_MAP_DIR,
        help="Folder containing incident_###.png terrain maps.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder where batch JSONL and manifest JSONL files are written.",
    )
    parser.add_argument(
        "--job-log",
        type=Path,
        default=DEFAULT_JOB_LOG,
        help="Append-only JSONL file for submitted batch job ids.",
    )
    parser.add_argument(
        "--rows",
        default=None,
        help=(
            '1-based CSV row selection, e.g. "1-30", "1", or "1,5,7". '
            "Defaults to the first 30 rows."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Deprecated compatibility option. Used only when --rows is omitted.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeated requests per incident and prompt variant.",
    )
    parser.add_argument(
        "--variants",
        default="all",
        help='Prompt family to include: "all", "vanilla", or "saliency".',
    )
    parser.add_argument(
        "--prompt-version",
        default="all",
        help='Prompt versions to include: "all", "1-3", "1,3", or "1".',
    )
    parser.add_argument(
        "--image-detail",
        choices=["low", "high", "original", "auto"],
        default="original",
        help="detail value for the input_image block.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Optional cap for total generated tokens. Does not set reasoning effort.",
    )
    parser.add_argument(
        "--completion-window",
        choices=["24h"],
        default="24h",
        help="OpenAI Batch API completion window.",
    )
    return parser.parse_args()


def parse_positive_int(value: str, label: str) -> int:
    try:
        number = int(value.removeprefix("v"))
    except ValueError as exc:
        raise ValueError(f"{label} must be a positive integer, got {value!r}") from exc

    if number <= 0:
        raise ValueError(f"{label} must be positive, got {value!r}")
    return number


def parse_number_spec(spec: str, label: str) -> list[int] | None:
    value = spec.strip().lower()
    if value == "all":
        return None
    if not value:
        raise ValueError(f"{label} cannot be empty")

    numbers: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"{label} contains an empty item, got {spec!r}")

        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = parse_positive_int(start_text.strip(), label)
            end = parse_positive_int(end_text.strip(), label)
            if end < start:
                raise ValueError(f"{label} range must be ascending, got {part!r}")
            numbers.extend(range(start, end + 1))
        else:
            numbers.append(parse_positive_int(part, label))

    if len(set(numbers)) != len(numbers):
        raise ValueError(f"{label} cannot contain duplicates, got {spec!r}")
    return numbers


def selected_row_spec(args: argparse.Namespace) -> str:
    if args.rows is not None:
        return args.rows
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    return f"1-{args.limit}"


def read_initial_conditions(path: Path, row_spec: str) -> list[InitialCondition]:
    row_numbers = parse_number_spec(row_spec, "row selection")
    all_rows: list[InitialCondition] = []

    with path.open(newline="", encoding="utf-8") as file:
        first_line = file.readline()
        if first_line.startswith("incident_index"):
            file.seek(0)

        reader = csv.DictReader(file)
        required_columns = {
            "incident_index",
            "IPP_lat",
            "IPP_lon",
            "find_lat",
            "find_lon",
        }
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{path} is missing required columns: {missing}")

        for row in reader:
            all_rows.append(
                InitialCondition(
                    incident_index=int(row["incident_index"]),
                    ipp_lat=float(row["IPP_lat"]),
                    ipp_lon=float(row["IPP_lon"]),
                    find_lat=float(row["find_lat"]),
                    find_lon=float(row["find_lon"]),
                )
            )

    if not all_rows:
        raise ValueError(f"No initial conditions found in {path}")

    if row_numbers is None:
        return all_rows

    max_row = len(all_rows)
    missing_rows = [row_number for row_number in row_numbers if row_number > max_row]
    if missing_rows:
        formatted = ", ".join(str(row_number) for row_number in missing_rows)
        raise ValueError(
            f"{path} has only {max_row} data rows; requested row(s): {formatted}"
        )

    rows = [all_rows[row_number - 1] for row_number in row_numbers]
    return rows


def load_prompt_variants(
    path: Path,
    variants: str,
    prompt_version: str,
) -> list[PromptVariant]:
    with path.open(encoding="utf-8") as file:
        raw_prompts = json.load(file)

    if not isinstance(raw_prompts, dict):
        raise ValueError(f"{path} must contain a JSON object")

    variant_value = variants.strip().lower()
    if variant_value == "all":
        selected_names = list(raw_prompts)
    elif variant_value in {"vanilla", "saliency"}:
        selected_names = [variant_value]
    else:
        raise ValueError('--variants must be one of: "all", "vanilla", "saliency"')

    requested_version_numbers = parse_number_spec(prompt_version, "prompt version")
    requested_versions = (
        None
        if requested_version_numbers is None
        else {f"v{number}" for number in requested_version_numbers}
    )
    prompt_variants: list[PromptVariant] = []

    for prompt_name in selected_names:
        if prompt_name not in raw_prompts:
            raise ValueError(f"Prompt name {prompt_name!r} was not found in {path}")

        versions = raw_prompts[prompt_name]
        if not isinstance(versions, dict):
            raise ValueError(f"{path} prompt {prompt_name!r} must be an object")

        missing_versions = set()
        if requested_versions is not None:
            missing_versions = requested_versions.difference(versions)
        if missing_versions:
            missing = ", ".join(sorted(missing_versions))
            raise ValueError(
                f"{path} prompt {prompt_name!r} is missing requested version(s): {missing}"
            )

        for version in versions:
            if requested_versions is not None and version not in requested_versions:
                continue

            prompt_text = versions[version]
            if not isinstance(prompt_text, str) or not prompt_text.strip():
                raise ValueError(
                    f"{path} prompt {prompt_name!r}/{version!r} must be a non-empty string"
                )

            prompt_variants.append(
                PromptVariant(name=prompt_name, version=version, text=prompt_text)
            )

    if not prompt_variants:
        raise ValueError("No prompt variants matched the requested filters")
    return prompt_variants


def map_path_for_incident(map_dir: Path, incident_index: int) -> Path:
    return map_dir / f"incident_{incident_index:03d}.png"


def upload_vision_files(
    client: OpenAI,
    conditions: Iterable[InitialCondition],
    map_dir: Path,
) -> dict[int, dict[str, str]]:
    uploads: dict[int, dict[str, str]] = {}
    for condition in conditions:
        image_path = map_path_for_incident(map_dir, condition.incident_index)
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing terrain map: {image_path}")

        with image_path.open("rb") as image_file:
            uploaded_file = client.files.create(file=image_file, purpose="vision")

        uploads[condition.incident_index] = {
            "file_id": uploaded_file.id,
            "path": str(image_path),
        }
        print(
            f"uploaded incident {condition.incident_index:03d}: "
            f"{image_path} -> {uploaded_file.id}"
        )

    return uploads


def build_request_body(
    *,
    model: str,
    prompt_text: str,
    file_id: str,
    image_detail: str,
    max_output_tokens: int | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt_text,
                    },
                    {
                        "type": "input_image",
                        "file_id": file_id,
                        "detail": image_detail,
                    },
                ],
            }
        ],
    }

    if max_output_tokens is not None:
        body["max_output_tokens"] = max_output_tokens

    return body


def custom_id(
    *,
    incident_index: int,
    prompt: PromptVariant,
    repeat: int,
) -> str:
    prompt_slug = f"{safe_path_part(prompt.name)}-{safe_path_part(prompt.version)}"
    return (
        f"sar__incident-{incident_index:03d}"
        f"__prompt-{prompt_slug}"
        f"__repeat-{repeat:02d}"
    )


def build_requests_and_manifest(
    *,
    args: argparse.Namespace,
    conditions: list[InitialCondition],
    prompts: list[PromptVariant],
    uploads: dict[int, dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")

    requests: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []

    for condition in conditions:
        upload = uploads[condition.incident_index]
        for prompt in prompts:
            for repeat in range(1, args.repeats + 1):
                request_id = custom_id(
                    incident_index=condition.incident_index,
                    prompt=prompt,
                    repeat=repeat,
                )
                body = build_request_body(
                    model=args.model,
                    prompt_text=prompt.text,
                    file_id=upload["file_id"],
                    image_detail=args.image_detail,
                    max_output_tokens=args.max_output_tokens,
                )
                requests.append(
                    {
                        "custom_id": request_id,
                        "method": "POST",
                        "url": RESPONSES_ENDPOINT,
                        "body": body,
                    }
                )
                manifest.append(
                    {
                        "custom_id": request_id,
                        "model": args.model,
                        "incident_index": condition.incident_index,
                        "ipp_lat": condition.ipp_lat,
                        "ipp_lon": condition.ipp_lon,
                        "find_lat": condition.find_lat,
                        "find_lon": condition.find_lon,
                        "prompt_name": prompt.name,
                        "prompt_version": prompt.version,
                        "repeat": repeat,
                        "map_path": upload["path"],
                        "vision_file_id": upload["file_id"],
                        "endpoint": RESPONSES_ENDPOINT,
                        "image_detail": args.image_detail,
                    }
                )

    return requests, manifest


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def append_job_log(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_run_context(args: argparse.Namespace) -> RunContext:
    run_timestamp = datetime.now(timezone.utc)
    run_id = run_timestamp.strftime("%Y%m%dT%H%M%SZ")
    model_slug = safe_path_part(args.model)
    run_slug = f"sar_{model_slug}_{run_id}"
    model_dir = args.output_dir / model_slug
    return RunContext(
        timestamp=run_timestamp,
        row_spec=selected_row_spec(args),
        model_slug=model_slug,
        run_slug=run_slug,
        batch_path=model_dir / f"{run_slug}.jsonl",
        manifest_path=model_dir / f"{run_slug}.manifest.jsonl",
    )


def load_batch_inputs(
    args: argparse.Namespace,
    context: RunContext,
) -> tuple[list[InitialCondition], list[PromptVariant]]:
    conditions = read_initial_conditions(args.dataset_csv, context.row_spec)
    prompts = load_prompt_variants(
        args.prompts,
        variants=args.variants,
        prompt_version=args.prompt_version,
    )
    return conditions, prompts


def write_batch_files(
    *,
    requests: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
    context: RunContext,
) -> int:
    request_count = write_jsonl(context.batch_path, requests)
    manifest_count = write_jsonl(context.manifest_path, manifest)
    if request_count != manifest_count:
        raise RuntimeError(
            f"Request count ({request_count}) does not match manifest count ({manifest_count})"
        )
    return request_count


def submit_batch_file(
    *,
    client: OpenAI,
    args: argparse.Namespace,
    context: RunContext,
    request_count: int,
) -> SubmittedBatch:
    with context.batch_path.open("rb") as batch_file:
        uploaded_batch_file = client.files.create(file=batch_file, purpose="batch")

    batch = client.batches.create(
        input_file_id=uploaded_batch_file.id,
        endpoint=RESPONSES_ENDPOINT,
        completion_window=args.completion_window,
        metadata={
            "source": "src/SAR/create_sar_batch.py",
            "model": args.model,
            "request_count": str(request_count),
        },
    )
    return SubmittedBatch(
        uploaded_batch_file=uploaded_batch_file,
        batch=batch,
        request_count=request_count,
    )


def build_job_record(
    *,
    args: argparse.Namespace,
    context: RunContext,
    conditions: list[InitialCondition],
    prompts: list[PromptVariant],
    uploads: dict[int, dict[str, str]],
    submitted: SubmittedBatch,
) -> dict[str, Any]:
    return {
        "timestamp_utc": context.timestamp.isoformat(),
        "batch_id": submitted.batch.id,
        "status": submitted.batch.status,
        "model": args.model,
        "endpoint": RESPONSES_ENDPOINT,
        "completion_window": args.completion_window,
        "request_count": submitted.request_count,
        "batch_input_file_id": submitted.uploaded_batch_file.id,
        "batch_input_path": str(context.batch_path),
        "manifest_path": str(context.manifest_path),
        "dataset_csv_path": str(args.dataset_csv),
        "prompt_path": str(args.prompts),
        "map_dir": str(args.map_dir),
        "rows": context.row_spec,
        "incident_count": len(conditions),
        "incident_indices": [condition.incident_index for condition in conditions],
        "repeats": args.repeats,
        "variants": args.variants,
        "prompt_version": args.prompt_version,
        "prompt_variants": [
            {"name": prompt.name, "version": prompt.version} for prompt in prompts
        ],
        "image_detail": args.image_detail,
        "vision_uploads": uploads,
        "uploaded_batch_file": object_to_dict(submitted.uploaded_batch_file),
        "batch": object_to_dict(submitted.batch),
    }


def print_submission_summary(
    *,
    args: argparse.Namespace,
    context: RunContext,
    submitted: SubmittedBatch,
) -> None:
    print(f"batch_id: {submitted.batch.id}")
    print(f"request_count: {submitted.request_count}")
    print(f"rows: {context.row_spec}")
    print(f"variants: {args.variants}")
    print(f"prompt_version: {args.prompt_version}")
    print(f"batch_input_path: {context.batch_path}")
    print(f"manifest_path: {context.manifest_path}")
    print(f"job_log: {args.job_log}")


def main() -> None:
    args = parse_args()
    context = build_run_context(args)
    conditions, prompts = load_batch_inputs(args, context)
    client = OpenAI(api_key=openai_api_key(None))
    uploads = upload_vision_files(client, conditions, args.map_dir)
    requests, manifest = build_requests_and_manifest(
        args=args,
        conditions=conditions,
        prompts=prompts,
        uploads=uploads,
    )
    request_count = write_batch_files(
        requests=requests,
        manifest=manifest,
        context=context,
    )
    submitted = submit_batch_file(
        client=client,
        args=args,
        context=context,
        request_count=request_count,
    )
    append_job_log(
        args.job_log,
        build_job_record(
            args=args,
            context=context,
            conditions=conditions,
            prompts=prompts,
            uploads=uploads,
            submitted=submitted,
        ),
    )
    print_submission_summary(args=args, context=context, submitted=submitted)


if __name__ == "__main__":
    main()
