import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse a raw OpenAI /v1/responses SAR batch output JSONL into a JSON "
            "summary with the textual response, reasoning, and chosen coordinate."
        )
    )
    parser.add_argument(
        "--response-file",
        type=Path,
        required=True,
        help="Raw downloaded OpenAI batch output JSONL.",
    )
    parser.add_argument(
        "--manifest-file",
        type=Path,
        required=True,
        help="Manifest JSONL written next to the submitted SAR batch request file.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Parsed JSON output path. Defaults to <response-file>.parsed.json.",
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
                raise ValueError(
                    f"{path}:{line_number} is invalid JSON: {exc}"
                ) from exc
    return rows


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    manifest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        custom_id = row.get("custom_id")
        if not custom_id:
            raise ValueError(f"{path} contains a manifest row without custom_id")
        if custom_id in manifest:
            raise ValueError(f"{path} contains duplicate custom_id: {custom_id}")
        manifest[custom_id] = row
    return manifest


def response_custom_id(row: dict[str, Any]) -> str:
    custom_id = row.get("custom_id")
    if not custom_id:
        raise ValueError(f"Response row is missing custom_id: {row}")
    return custom_id


def response_body(row: dict[str, Any]) -> dict[str, Any] | None:
    response = row.get("response") or {}
    return response.get("body")


def response_error(row: dict[str, Any]) -> Any:
    response = row.get("response") or {}
    return row.get("error") or response.get("error")


def iter_response_output(body: dict[str, Any]) -> Iterable[dict[str, Any]]:
    output = body.get("output") or []
    if not isinstance(output, list):
        return []
    return output


def collect_output_text(body: dict[str, Any]) -> str:
    texts: list[str] = []
    for item in iter_response_output(body):
        if item.get("type") == "message":
            for content in item.get("content") or []:
                if content.get("type") in {"output_text", "text"} and content.get(
                    "text"
                ):
                    texts.append(content["text"])
        elif item.get("type") in {"output_text", "text"} and item.get("text"):
            texts.append(item["text"])
    return "\n".join(texts).strip()


def parse_tag(text: str, tag: str) -> str | None:
    match = re.search(
        rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.IGNORECASE | re.DOTALL
    )
    if not match:
        return None
    return match.group(1).strip()


def parse_coordinate(text: str) -> dict[str, float] | None:
    coord_text = parse_tag(text, "coord")
    if coord_text is None:
        return None

    match = re.fullmatch(
        r"\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*",
        coord_text,
    )
    if match is None:
        values = re.findall(r"[-+]?\d*\.?\d+", coord_text)
        if len(values) < 2:
            return None
        x_text, y_text = values[:2]
    else:
        x_text, y_text = match.groups()

    try:
        return {"x": float(x_text), "y": float(y_text)}
    except ValueError:
        return None


def prompt_key(meta: dict[str, Any]) -> str:
    return (
        f"{meta.get('prompt_name', 'unknown')}:{meta.get('prompt_version', 'unknown')}"
    )


def parse_response_rows(
    *,
    response_rows: list[dict[str, Any]],
    manifest: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    prompts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failures: list[dict[str, Any]] = []

    for row in response_rows:
        custom_id = response_custom_id(row)
        meta = manifest.get(custom_id)
        if meta is None:
            failures.append(
                {
                    "custom_id": custom_id,
                    "error": "Response custom_id not found in manifest.",
                }
            )
            continue

        error = response_error(row)
        body = response_body(row)
        if error or body is None:
            failures.append(
                {
                    "custom_id": custom_id,
                    "manifest": meta,
                    "error": error or "Response body missing.",
                }
            )
            continue

        full_text = collect_output_text(body)
        entry = {
            "custom_id": custom_id,
            "incident_index": meta.get("incident_index"),
            "repeat": meta.get("repeat"),
            "prompt_name": meta.get("prompt_name"),
            "prompt_version": meta.get("prompt_version"),
            "full_text": full_text,
            "reasoning": parse_tag(full_text, "reasoning"),
            "coordinate": parse_coordinate(full_text),
            "manifest": meta,
        }
        prompts[prompt_key(meta)].append(entry)

    return {
        "prompts": dict(sorted(prompts.items())),
        "failures": failures,
        "summary": {
            "response_rows": len(response_rows),
            "manifest_rows": len(manifest),
            "parsed_rows": sum(len(entries) for entries in prompts.values()),
            "failure_rows": len(failures),
        },
    }


def main() -> None:
    args = parse_args()
    output_json = args.output_json
    if output_json is None:
        output_json = args.response_file.with_suffix(".parsed.json")

    manifest = load_manifest(args.manifest_file)
    response_rows = read_jsonl(args.response_file)
    parsed = parse_response_rows(
        response_rows=response_rows,
        manifest=manifest,
    )
    parsed["source"] = {
        "response_file": str(args.response_file),
        "manifest_file": str(args.manifest_file),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as file:
        json.dump(parsed, file, indent=2, ensure_ascii=False)

    print(f"parsed_json: {output_json}")
    print(json.dumps(parsed["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
