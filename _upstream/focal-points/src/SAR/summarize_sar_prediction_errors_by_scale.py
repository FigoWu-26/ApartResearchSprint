import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

try:
    from .summarize_sar_prediction_errors import (
        DEFAULT_GROUND_TRUTH_CSV,
        DEFAULT_METHODS,
        collect_predictions,
        expand_parsed_paths,
        load_ground_truth,
    )
except ImportError:
    from summarize_sar_prediction_errors import (
        DEFAULT_GROUND_TRUTH_CSV,
        DEFAULT_METHODS,
        collect_predictions,
        expand_parsed_paths,
        load_ground_truth,
    )


DEFAULT_OUTPUT_CSV = Path(
    "data/SAR_prompts/batch_responses/openai/gpt-5.5/"
    "q1_q2-q30_part1_error_summary_by_scale.csv"
)


@dataclass(frozen=True)
class ScaleInfo:
    incident_index: int
    meters_per_cell: float
    ground_width_m: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize SAR prediction errors by map scale. Accepts one or more "
            "parsed JSON files produced by parse_sar_batch_response.py, or "
            "directories containing *.parsed.json files."
        )
    )
    parser.add_argument(
        "--parsed-json",
        nargs="+",
        type=Path,
        required=True,
        help="One or more parsed SAR response JSON files or directories.",
    )
    parser.add_argument(
        "--ground-truth-csv",
        type=Path,
        default=DEFAULT_GROUND_TRUTH_CSV,
        help=(
            "CSV with incident_index, find_x_m, find_y_m, meters_per_cell, "
            f"and ground_width_m. Defaults to {DEFAULT_GROUND_TRUTH_CSV}."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"By-scale summary CSV output path. Defaults to {DEFAULT_OUTPUT_CSV}.",
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


def load_scale_info(path: Path) -> dict[int, ScaleInfo]:
    scale_info: dict[int, ScaleInfo] = {}
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"incident_index", "meters_per_cell", "ground_width_m"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

        for row in reader:
            incident_index = int(row["incident_index"])
            if incident_index in scale_info:
                raise ValueError(
                    f"{path} has duplicate incident_index {incident_index}"
                )
            scale_info[incident_index] = ScaleInfo(
                incident_index=incident_index,
                meters_per_cell=float(row["meters_per_cell"]),
                ground_width_m=float(row["ground_width_m"]),
            )

    if not scale_info:
        raise ValueError(f"No scale rows found in {path}")
    return scale_info


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.3f}"


def format_scale(value: float) -> str:
    return f"{value:.9f}".rstrip("0").rstrip(".")


def stats(values: list[float]) -> tuple[str, str]:
    if not values:
        return "", ""

    average = sum(values) / len(values)
    squared_spread = sum((value - average) ** 2 for value in values) / len(values)
    stddev = math.sqrt(squared_spread)
    return format_float(average), format_float(stddev)


def output_fieldnames(methods: list[str]) -> list[str]:
    fieldnames = [
        "meters_per_cell",
        "ground_width_m",
        "incident_count",
    ]
    for method in methods:
        fieldnames.extend(
            [
                f"{method}_prediction_count",
                f"{method}_incident_count",
                f"{method}_avg_distance_m",
                f"{method}_stddev_distance_m",
            ]
        )
    return fieldnames


def build_scale_rows(
    *,
    predictions,
    scale_info: dict[int, ScaleInfo],
    methods: list[str],
) -> list[dict[str, str]]:
    by_scale_method: dict[tuple[float, float, str], list[float]] = defaultdict(list)
    incidents_by_scale: dict[tuple[float, float], set[int]] = defaultdict(set)
    method_incidents_by_scale: dict[tuple[float, float, str], set[int]] = defaultdict(
        set
    )

    for prediction in predictions:
        info = scale_info.get(prediction.incident_index)
        if info is None:
            raise ValueError(
                "Missing scale info for incident "
                f"{prediction.incident_index} from {prediction.parsed_file}"
            )

        scale_key = (info.meters_per_cell, info.ground_width_m)
        method_key = (*scale_key, prediction.method)
        by_scale_method[method_key].append(prediction.distance_m)
        incidents_by_scale[scale_key].add(prediction.incident_index)
        method_incidents_by_scale[method_key].add(prediction.incident_index)

    rows: list[dict[str, str]] = []
    for meters_per_cell, ground_width_m in sorted(incidents_by_scale):
        scale_key = (meters_per_cell, ground_width_m)
        row = {
            "meters_per_cell": format_scale(meters_per_cell),
            "ground_width_m": format_float(ground_width_m),
            "incident_count": str(len(incidents_by_scale[scale_key])),
        }
        for method in methods:
            method_key = (*scale_key, method)
            values = by_scale_method.get(method_key, [])
            average, stddev = stats(values)
            row[f"{method}_prediction_count"] = str(len(values))
            row[f"{method}_incident_count"] = str(
                len(method_incidents_by_scale.get(method_key, set()))
            )
            row[f"{method}_avg_distance_m"] = average
            row[f"{method}_stddev_distance_m"] = stddev
        rows.append(row)

    return rows


def write_csv(path: Path, rows: list[dict[str, str]], methods: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=output_fieldnames(methods))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    methods = list(dict.fromkeys(args.methods))
    parsed_paths = expand_parsed_paths(args.parsed_json)
    ground_truth = load_ground_truth(args.ground_truth_csv)
    scale_info = load_scale_info(args.ground_truth_csv)
    predictions, skipped = collect_predictions(
        parsed_paths=parsed_paths,
        ground_truth=ground_truth,
        methods=set(methods),
        allow_duplicate_custom_ids=args.allow_duplicate_custom_ids,
        include_prompt_version=args.include_prompt_version,
    )

    if not predictions:
        raise ValueError("No predictions with coordinates matched the selected methods")

    rows = build_scale_rows(
        predictions=predictions,
        scale_info=scale_info,
        methods=methods,
    )
    write_csv(args.output_csv, rows, methods)

    print(f"parsed_files: {len(parsed_paths)}")
    print(f"predictions: {len(predictions)}")
    print(f"skipped_entries: {len(skipped)}")
    print(f"scale_groups: {len(rows)}")
    print(f"summary_csv: {args.output_csv}")


if __name__ == "__main__":
    main()
