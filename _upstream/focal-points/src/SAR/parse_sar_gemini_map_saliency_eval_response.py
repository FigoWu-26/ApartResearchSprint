import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

CUSTOM_ID_RE = re.compile(
    r"^sar_map_saliency_eval__incident-(?P<incident>\d{3})__image-(?P<image_version>v[12])$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse downloaded Gemini Batch API JSONL responses for the SAR map "
            "saliency evaluation prompts."
        )
    )
    parser.add_argument(
        "--response-file",
        type=Path,
        required=True,
        help="Downloaded Gemini batch response JSONL.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Parsed JSON output path. Defaults to <response-file>.parsed.json.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional flat CSV output path.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is invalid JSON: {exc}") from exc
    return rows


def parse_custom_id(custom_id: str) -> dict[str, Any]:
    match = CUSTOM_ID_RE.match(custom_id)
    if not match:
        return {
            "incident": None,
            "incident_index": None,
            "image_version": None,
            "custom_id_parse_error": "Unexpected custom_id format.",
        }

    incident_index = int(match.group("incident"))
    return {
        "incident": f"incident_{incident_index:03d}",
        "incident_index": incident_index,
        "image_version": match.group("image_version"),
        "custom_id_parse_error": None,
    }


def parse_tag(text: str, tag: str) -> str | None:
    match = re.search(
        rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    return match.group(1).strip()


def normalize_answer(answer: str | None) -> str | None:
    if answer is None:
        return None
    match = re.search(r"\b(yes|no)\b", answer, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).lower()


def gemini_response_text(row: dict[str, Any]) -> str | None:
    response = row.get("response")
    if not isinstance(response, dict):
        return None
    candidates = response.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return None
    content = candidates[0].get("content")
    if not isinstance(content, dict):
        return None
    parts = content.get("parts")
    if not isinstance(parts, list):
        return None
    texts = [
        part["text"]
        for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    ]
    return "".join(texts) if texts else None


def gemini_error(row: dict[str, Any]) -> Any:
    if row.get("error"):
        return row["error"]
    response = row.get("response")
    if isinstance(response, dict) and response.get("error"):
        return response["error"]
    status = row.get("status")
    if isinstance(status, dict) and status.get("error"):
        return status["error"]
    return None


def parse_response_rows(response_rows: list[dict[str, Any]]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for row in response_rows:
        custom_id = row.get("key")
        if not isinstance(custom_id, str):
            failures.append(
                {
                    "custom_id": None,
                    "incident": None,
                    "incident_index": None,
                    "image_version": None,
                    "custom_id_parse_error": "Missing Gemini response key.",
                    "error": "Missing Gemini response key.",
                    "row": row,
                }
            )
            continue

        id_meta = parse_custom_id(custom_id)
        error = gemini_error(row)
        full_response = gemini_response_text(row)
        if error or full_response is None:
            failures.append(
                {
                    "custom_id": custom_id,
                    **id_meta,
                    "error": error or "Gemini response text missing.",
                    "row": row,
                }
            )
            continue

        answer = parse_tag(full_response, "answer")
        reasoning = parse_tag(full_response, "reasoning")
        records.append(
            {
                "custom_id": custom_id,
                **id_meta,
                "full_response": full_response,
                "reasoning": reasoning,
                "answer": answer,
                "answer_normalized": normalize_answer(answer),
            }
        )

    answer_counts = Counter(
        record["answer_normalized"] or "missing" for record in records
    )
    image_version_counts = Counter(
        record["image_version"] or "unknown" for record in records
    )

    return {
        "records": records,
        "failures": failures,
        "summary": {
            "response_rows": len(response_rows),
            "parsed_rows": len(records),
            "failure_rows": len(failures),
            "answer_counts": dict(sorted(answer_counts.items())),
            "image_version_counts": dict(sorted(image_version_counts.items())),
            "custom_id_parse_failures": sum(
                1 for record in records if record["custom_id_parse_error"]
            )
            + sum(1 for failure in failures if failure["custom_id_parse_error"]),
        },
    }


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "custom_id",
        "incident",
        "incident_index",
        "image_version",
        "answer",
        "answer_normalized",
        "reasoning",
        "full_response",
        "custom_id_parse_error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    output_json = args.output_json or args.response_file.with_suffix(".parsed.json")

    response_rows = read_jsonl(args.response_file)
    parsed = parse_response_rows(response_rows)
    parsed["source"] = {
        "response_file": str(args.response_file),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as file:
        json.dump(parsed, file, indent=2, ensure_ascii=False)

    if args.output_csv is not None:
        write_csv(args.output_csv, parsed["records"])

    print(f"parsed_json: {output_json}")
    if args.output_csv is not None:
        print(f"parsed_csv: {args.output_csv}")
    print(json.dumps(parsed["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
