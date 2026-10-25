import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_GROUND_TRUTH_CSV = Path("data/SAR_prompts/find_location_ground_truth.csv")
DEFAULT_OUTPUT_CSV = Path("data/SAR_prompts/sar_prediction_error_summary.csv")
DEFAULT_METHODS = ("vanilla", "saliency")


@dataclass(frozen=True)
class GroundTruth:
    incident_index: int
    x_m: float
    y_m: float


@dataclass(frozen=True)
class Prediction:
    parsed_file: Path
    custom_id: str
    incident_index: int
    method: str
    repeat: int | None
    x_m: float
    y_m: float
    distance_m: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize SAR model coordinate errors against find-location ground truth. "
            "Accepts one or more parsed JSON files produced by parse_sar_batch_response.py."
        )
    )
    parser.add_argument(
        "--parsed-json",
        nargs="+",
        type=Path,
        required=True,
        help=(
            "One or more parsed SAR response JSON files. Directories are expanded to "
            "*.parsed.json files inside that directory."
        ),
    )
    parser.add_argument(
        "--ground-truth-csv",
        type=Path,
        default=DEFAULT_GROUND_TRUTH_CSV,
        help=(
            "CSV with incident_index, find_x_m, and find_y_m columns. "
            f"Defaults to {DEFAULT_GROUND_TRUTH_CSV}."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Summary CSV output path. Defaults to {DEFAULT_OUTPUT_CSV}.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
        help='Prompt methods to summarize. Defaults to "vanilla saliency".',
    )
    parser.add_argument(
        "--allow-duplicate-custom-ids",
        action="store_true",
        help="Allow duplicate custom_id values across parsed files.",
    )
    parser.add_argument(
        "--include-prompt-version",
        action="store_true",
        help=(
            "Group methods by prompt name and version, e.g. vanilla_v1 and "
            "saliency_v2, instead of only vanilla and saliency."
        ),
    )
    return parser.parse_args()


def expand_parsed_paths(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(sorted(path.glob("*.parsed.json")))
        else:
            expanded.append(path)

    missing = [str(path) for path in expanded if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing parsed JSON file(s):\n" + "\n".join(missing))
    if not expanded:
        raise ValueError("No parsed JSON files found")
    return expanded


def load_ground_truth(path: Path) -> dict[int, GroundTruth]:
    ground_truth: dict[int, GroundTruth] = {}
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"incident_index", "find_x_m", "find_y_m"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

        for row in reader:
            incident_index = int(row["incident_index"])
            if incident_index in ground_truth:
                raise ValueError(
                    f"{path} has duplicate incident_index {incident_index}"
                )
            ground_truth[incident_index] = GroundTruth(
                incident_index=incident_index,
                x_m=float(row["find_x_m"]),
                y_m=float(row["find_y_m"]),
            )

    if not ground_truth:
        raise ValueError(f"No ground-truth rows found in {path}")
    return ground_truth


def load_parsed_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def iter_parsed_entries(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    prompts = parsed.get("prompts")
    if not isinstance(prompts, dict):
        raise ValueError("Parsed JSON is missing a prompts object")

    entries: list[dict[str, Any]] = []
    for prompt_entries in prompts.values():
        if not isinstance(prompt_entries, list):
            continue
        entries.extend(entry for entry in prompt_entries if isinstance(entry, dict))
    return entries


def entry_prompt_version(entry: dict[str, Any]) -> str | None:
    prompt_version = entry.get("prompt_version")
    if isinstance(prompt_version, str) and prompt_version:
        return prompt_version

    manifest = entry.get("manifest")
    if isinstance(manifest, dict):
        manifest_prompt_version = manifest.get("prompt_version")
        if isinstance(manifest_prompt_version, str) and manifest_prompt_version:
            return manifest_prompt_version

    return None


def entry_method(entry: dict[str, Any], include_prompt_version: bool = False) -> str:
    prompt_name = entry.get("prompt_name")
    if isinstance(prompt_name, str) and prompt_name:
        method = prompt_name
    else:
        manifest = entry.get("manifest")
        if isinstance(manifest, dict):
            manifest_prompt_name = manifest.get("prompt_name")
            if isinstance(manifest_prompt_name, str) and manifest_prompt_name:
                method = manifest_prompt_name
            else:
                method = "unknown"
        else:
            method = "unknown"

    if include_prompt_version:
        prompt_version = entry_prompt_version(entry)
        if prompt_version:
            return f"{method}_{prompt_version}"

    return method


def entry_incident_index(entry: dict[str, Any]) -> int:
    incident_index = entry.get("incident_index")
    if incident_index is None and isinstance(entry.get("manifest"), dict):
        incident_index = entry["manifest"].get("incident_index")
    if incident_index is None:
        raise ValueError(
            f"Parsed entry is missing incident_index: {entry.get('custom_id')}"
        )
    return int(incident_index)


def entry_coordinate(entry: dict[str, Any]) -> tuple[float, float] | None:
    coordinate = entry.get("coordinate")
    if not isinstance(coordinate, dict):
        return None
    if coordinate.get("x") is None or coordinate.get("y") is None:
        return None
    return float(coordinate["x"]), float(coordinate["y"])


def collect_predictions(
    *,
    parsed_paths: list[Path],
    ground_truth: dict[int, GroundTruth],
    methods: set[str],
    allow_duplicate_custom_ids: bool,
    include_prompt_version: bool = False,
) -> tuple[list[Prediction], list[dict[str, str]]]:
    predictions: list[Prediction] = []
    skipped: list[dict[str, str]] = []
    seen_custom_ids: set[str] = set()

    for parsed_path in parsed_paths:
        parsed = load_parsed_json(parsed_path)
        for entry in iter_parsed_entries(parsed):
            custom_id = str(entry.get("custom_id") or "")
            if (
                custom_id
                and custom_id in seen_custom_ids
                and not allow_duplicate_custom_ids
            ):
                raise ValueError(
                    f"Duplicate custom_id across parsed files: {custom_id}. "
                    "Pass --allow-duplicate-custom-ids to include duplicates."
                )
            if custom_id:
                seen_custom_ids.add(custom_id)

            method = entry_method(
                entry,
                include_prompt_version=include_prompt_version,
            )
            if method not in methods:
                skipped.append(
                    {
                        "parsed_file": str(parsed_path),
                        "custom_id": custom_id,
                        "reason": f"method {method!r} not selected",
                    }
                )
                continue

            coordinate = entry_coordinate(entry)
            if coordinate is None:
                skipped.append(
                    {
                        "parsed_file": str(parsed_path),
                        "custom_id": custom_id,
                        "reason": "missing parsed coordinate",
                    }
                )
                continue

            incident_index = entry_incident_index(entry)
            truth = ground_truth.get(incident_index)
            if truth is None:
                skipped.append(
                    {
                        "parsed_file": str(parsed_path),
                        "custom_id": custom_id,
                        "reason": f"missing ground truth for incident {incident_index}",
                    }
                )
                continue

            x_m, y_m = coordinate
            distance_m = math.hypot(x_m - truth.x_m, y_m - truth.y_m)
            repeat = entry.get("repeat")
            predictions.append(
                Prediction(
                    parsed_file=parsed_path,
                    custom_id=custom_id,
                    incident_index=incident_index,
                    method=method,
                    repeat=int(repeat) if repeat is not None else None,
                    x_m=x_m,
                    y_m=y_m,
                    distance_m=distance_m,
                )
            )

    return predictions, skipped


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def stddev(values: list[float]) -> float | None:
    if not values:
        return None
    avg = sum(values) / len(values)
    spread = sum((value - avg) ** 2 for value in values) / len(values)
    return math.sqrt(spread)


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.3f}"


def method_stats(predictions: list[Prediction]) -> dict[str, str]:
    distances = [prediction.distance_m for prediction in predictions]
    avg = mean(distances)
    return {
        "prediction_count": str(len(predictions)),
        "avg_distance_m": format_float(avg),
        "stddev_distance_m": format_float(stddev(distances)),
    }


def build_summary_rows(
    *,
    predictions: list[Prediction],
    ground_truth: dict[int, GroundTruth],
    methods: list[str],
) -> list[dict[str, str]]:
    by_incident_method: dict[tuple[int, str], list[Prediction]] = defaultdict(list)
    by_method: dict[str, list[Prediction]] = defaultdict(list)
    incident_indices: set[int] = set()

    for prediction in predictions:
        by_incident_method[(prediction.incident_index, prediction.method)].append(
            prediction
        )
        by_method[prediction.method].append(prediction)
        incident_indices.add(prediction.incident_index)

    rows: list[dict[str, str]] = []
    for incident_index in sorted(incident_indices):
        truth = ground_truth[incident_index]
        row = {
            "scope": "incident",
            "incident_index": str(incident_index),
            "incident_count": "1",
            "ground_truth_x_m": format_float(truth.x_m),
            "ground_truth_y_m": format_float(truth.y_m),
        }
        for method in methods:
            stats = method_stats(by_incident_method.get((incident_index, method), []))
            for key, value in stats.items():
                row[f"{method}_{key}"] = value
        rows.append(row)

    overall_row = {
        "scope": "overall",
        "incident_index": "",
        "incident_count": str(len(incident_indices)),
        "ground_truth_x_m": "",
        "ground_truth_y_m": "",
    }
    for method in methods:
        stats = method_stats(by_method.get(method, []))
        for key, value in stats.items():
            overall_row[f"{method}_{key}"] = value
    rows.append(overall_row)
    return rows


def fieldnames_for_methods(methods: list[str]) -> list[str]:
    fieldnames = [
        "scope",
        "incident_index",
        "incident_count",
        "ground_truth_x_m",
        "ground_truth_y_m",
    ]
    for method in methods:
        fieldnames.extend(
            [
                f"{method}_prediction_count",
                f"{method}_avg_distance_m",
                f"{method}_stddev_distance_m",
            ]
        )
    return fieldnames


def write_summary_csv(
    path: Path, rows: list[dict[str, str]], methods: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fieldnames_for_methods(methods)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    methods = list(dict.fromkeys(args.methods))
    parsed_paths = expand_parsed_paths(args.parsed_json)
    ground_truth = load_ground_truth(args.ground_truth_csv)
    predictions, skipped = collect_predictions(
        parsed_paths=parsed_paths,
        ground_truth=ground_truth,
        methods=set(methods),
        allow_duplicate_custom_ids=args.allow_duplicate_custom_ids,
        include_prompt_version=args.include_prompt_version,
    )

    if not predictions:
        raise ValueError("No predictions with coordinates matched the selected methods")

    rows = build_summary_rows(
        predictions=predictions,
        ground_truth=ground_truth,
        methods=methods,
    )
    write_summary_csv(args.output_csv, rows, methods)

    print(f"parsed_files: {len(parsed_paths)}")
    print(f"predictions: {len(predictions)}")
    print(f"skipped_entries: {len(skipped)}")
    print(f"summary_csv: {args.output_csv}")


if __name__ == "__main__":
    main()
