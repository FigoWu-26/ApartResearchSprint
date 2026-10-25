import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any




@dataclass(frozen=True)
class JobRecord:
    batch_file: str | None
    provider: str | None
    model: str | None
    job_name: str


def normalize_provider(provider: str | None) -> str | None:
    if provider is None:
        return None

    value = provider.strip().lower()
    if value in {"gpt", "openai"}:
        return "openai"
    if value in {"gemini", "google"}:
        return "gemini"
    raise ValueError("provider must be one of: gpt, openai, gemini")


def safe_path_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "unknown"


def find_batch_file(filename: str, search_root: Path) -> Path:
    direct_path = Path(filename)
    if direct_path.is_file():
        return direct_path.resolve()

    matches = sorted(search_root.rglob(filename))
    matches = [path for path in matches if path.is_file()]

    if not matches:
        raise FileNotFoundError(f"Could not find {filename!r} under {search_root}")
    if len(matches) > 1:
        formatted = "\n".join(f"  - {path}" for path in matches)
        raise ValueError(
            f"Found multiple files named {filename!r}; pass a more specific path:\n{formatted}"
        )

    return matches[0]


def validate_openai_batch_model(path: Path, model: str) -> None:
    mismatches: list[tuple[int, str | None]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row_model = row.get("body", {}).get("model")
            if row_model != model:
                mismatches.append((line_number, row_model))
                if len(mismatches) >= 5:
                    break

    if mismatches:
        details = ", ".join(
            f"line {line_number}: {row_model!r}"
            for line_number, row_model in mismatches
        )
        raise ValueError(
            f"{path} contains OpenAI requests for a different model than {model!r}: {details}"
        )


def object_to_dict(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", by_alias=True, exclude_none=True)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return value
    return repr(value)


def append_or_replace_job_record(
    job_file: Path,
    batch_file: Path,
    provider: str,
    model: str,
    job_name: str,
) -> None:
    records = read_job_records(job_file) if job_file.exists() else []
    new_record = JobRecord(
        batch_file=batch_file.name,
        provider=provider,
        model=model,
        job_name=job_name,
    )

    replaced = False
    updated_records: list[JobRecord] = []
    for record in records:
        if record.batch_file == batch_file.name:
            updated_records.append(new_record)
            replaced = True
        else:
            updated_records.append(record)

    if not replaced:
        updated_records.append(new_record)

    job_file.parent.mkdir(parents=True, exist_ok=True)
    with open(job_file, "w", encoding="utf-8") as f:
        for record in updated_records:
            f.write(json.dumps(record.__dict__, ensure_ascii=False) + "\n")


def read_job_records(job_file: Path) -> list[JobRecord]:
    records: list[JobRecord] = []
    with open(job_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                records.append(
                    JobRecord(
                        batch_file=None,
                        provider=None,
                        model=None,
                        job_name=line,
                    )
                )
                continue

            records.append(
                JobRecord(
                    batch_file=data.get("batch_file"),
                    provider=normalize_provider(data.get("provider")),
                    model=data.get("model"),
                    job_name=data["job_name"],
                )
            )

    return records


def resolve_job_record(
    *,
    job_name: str | None,
    job_file: Path | None,
    batch_filename: str | None,
    provider: str | None,
    model: str | None,
) -> JobRecord:
    normalized_provider = normalize_provider(provider)

    if job_name:
        if normalized_provider is None:
            raise ValueError("--provider is required when --job-name is used")
        return JobRecord(
            batch_file=batch_filename,
            provider=normalized_provider,
            model=model,
            job_name=job_name,
        )

    if job_file is None:
        raise ValueError("Provide --job-name or --job-file")

    records = read_job_records(job_file)
    if batch_filename is not None:
        records = [record for record in records if record.batch_file == batch_filename]

    if len(records) != 1:
        choices = "\n".join(
            f"  - batch_file={record.batch_file!r}, provider={record.provider!r}, "
            f"model={record.model!r}, job_name={record.job_name!r}"
            for record in records
        )
        suffix = f"\n{choices}" if choices else ""
        raise ValueError(
            "Could not choose exactly one job record. Pass --batch-filename "
            f"or --job-name explicitly.{suffix}"
        )

    record = records[0]
    return JobRecord(
        batch_file=record.batch_file,
        provider=record.provider or normalized_provider,
        model=record.model or model,
        job_name=record.job_name,
    )


def openai_api_key(api_key: str | None) -> str:
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. This OpenAI Batch-API path is OpenAI-only (optional; the "
            "hackathon default is Nebius). Set OPENAI_API_KEY to use it."
        )
    return key


def gemini_api_key(api_key: str | None) -> str:
    # GEMINI_API_KEY and GOOGLE_API_KEY are two accepted names for the SAME Gemini key (not a swap).
    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "No Gemini key set (GEMINI_API_KEY or GOOGLE_API_KEY). This Gemini Batch-API path is "
            "Gemini-only (optional; the hackathon default is Nebius). Set one to use it."
        )
    return key
