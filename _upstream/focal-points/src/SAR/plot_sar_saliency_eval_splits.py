import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.SAR.plot_sar_error_by_scale import (
    ScaleMethodStats,
    ScaleSummary,
    save_scale_plot,
)
from src.SAR.plot_sar_prediction_errors import (
    MetricStats,
    combine_stats,
    fmt,
    merge_incident_summaries,
    overall_summary,
    read_summary_csv,
    save_bar_chart,
    save_ground_truth_plot,
    save_latex_table,
    save_line_chart,
    save_png_table,
    table_rows_for_incidents,
    table_rows_for_summary,
    write_combined_csv,
)
from src.SAR.summarize_sar_prediction_errors_by_scale import (
    format_float,
    format_scale,
    load_scale_info,
)

DEFAULT_PARSED_EVAL_JSON = Path(
    "data/SAR_prompts/batch_responses/openai/gpt-5.5/"
    "q1-q65_map-saliency-eval/"
    "batch_6a09e37a6d908190a17e6be2559f9835.parsed.json"
)
DEFAULT_COMBINED_SUMMARY_CSV = Path(
    "data/SAR_graphs/openai/gpt-5.5/q1-q65/shared/combined_error_summary.csv"
)
DEFAULT_OUTPUT_ROOT = Path("data/SAR_graphs/openai/gpt-5.5/q1-q65")
DEFAULT_GROUND_TRUTH_CSV = Path("data/SAR_prompts/find_location_ground_truth.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create shared-style SAR error plots split by map saliency-evaluation "
            "yes/no answers for v1 and v2 maps."
        )
    )
    parser.add_argument(
        "--saliency-eval-parsed-json",
        type=Path,
        default=DEFAULT_PARSED_EVAL_JSON,
        help=f"Parsed saliency-evaluation JSON. Defaults to {DEFAULT_PARSED_EVAL_JSON}.",
    )
    parser.add_argument(
        "--combined-summary-csv",
        type=Path,
        default=DEFAULT_COMBINED_SUMMARY_CSV,
        help=f"Combined SAR error summary CSV. Defaults to {DEFAULT_COMBINED_SUMMARY_CSV}.",
    )
    parser.add_argument(
        "--ground-truth-csv",
        type=Path,
        default=DEFAULT_GROUND_TRUTH_CSV,
        help=f"Ground-truth CSV with map scale columns. Defaults to {DEFAULT_GROUND_TRUTH_CSV}.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Folder that contains the existing shared graph folder. Defaults to {DEFAULT_OUTPUT_ROOT}.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="PNG output DPI.",
    )
    return parser.parse_args()


def infer_methods(summary_csv: Path) -> list[str]:
    with summary_csv.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []

    methods = [
        field[: -len("_prediction_count")]
        for field in fieldnames
        if field.endswith("_prediction_count")
    ]
    if not methods:
        raise ValueError(f"No *_prediction_count columns found in {summary_csv}")
    return methods


def load_answer_incidents(
    path: Path,
) -> tuple[dict[str, dict[str, set[int]]], dict[str, Counter[str]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("records")
    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a records list")

    incidents_by_version_answer: dict[str, dict[str, set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )
    answer_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for record in records:
        if not isinstance(record, dict):
            continue
        image_version = record.get("image_version")
        answer = record.get("answer_normalized")
        incident_index = record.get("incident_index")
        if image_version not in {"v1", "v2"}:
            continue
        if answer not in {"yes", "no"}:
            answer = "missing"
        if not isinstance(incident_index, int):
            continue

        incidents_by_version_answer[image_version][answer].add(incident_index)
        answer_counts[image_version][answer] += 1

    return incidents_by_version_answer, answer_counts


def write_summary_outputs_to_dir(
    *,
    output_dir: Path,
    incident_summaries,
    methods: list[str],
    dpi: int,
    title_suffix: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    overall = overall_summary(incident_summaries, methods)

    write_combined_csv(
        output_dir / "combined_error_summary.csv",
        incident_summaries,
        overall,
        methods,
    )
    save_line_chart(
        incident_summaries=incident_summaries,
        methods=methods,
        metric="mean",
        title=f"Average distance from ground truth by incident ({title_suffix})",
        ylabel="Average distance (m)",
        output_path=output_dir / "average_distance_by_incident.png",
        dpi=dpi,
    )
    save_line_chart(
        incident_summaries=incident_summaries,
        methods=methods,
        metric="stddev",
        title=f"Distance standard deviation by incident ({title_suffix})",
        ylabel="Standard deviation (m)",
        output_path=output_dir / "stddev_by_incident.png",
        dpi=dpi,
    )
    save_ground_truth_plot(
        incident_summaries=incident_summaries,
        output_path=output_dir / "ground_truth_locations.png",
        dpi=dpi,
        title=f"Ground-truth find coordinates ({title_suffix})",
    )
    save_bar_chart(
        labels=methods,
        values=[
            overall.methods.get(method, MetricStats(0, None, None)).mean
            for method in methods
        ],
        title=f"Overall average distance from ground truth ({title_suffix})",
        ylabel="Average distance (m)",
        output_path=output_dir / "overall_average_distance.png",
        dpi=dpi,
    )
    save_bar_chart(
        labels=methods,
        values=[
            overall.methods.get(method, MetricStats(0, None, None)).stddev
            for method in methods
        ],
        title=f"Overall distance standard deviation ({title_suffix})",
        ylabel="Standard deviation (m)",
        output_path=output_dir / "overall_stddev.png",
        dpi=dpi,
    )

    overall_rows = table_rows_for_summary(overall, methods)
    save_png_table(
        output_dir / "overall_table.png",
        overall_rows,
        f"Overall SAR prediction error metrics ({title_suffix})",
        dpi,
    )
    save_latex_table(
        output_dir / "overall_table.tex",
        overall_rows,
        f"Overall SAR prediction error metrics ({title_suffix})",
        f"tab:sar-overall-errors-{title_suffix.lower().replace(' ', '-')}",
    )

    incident_rows = table_rows_for_incidents(incident_summaries, methods)
    save_png_table(
        output_dir / "incident_metrics_table.png",
        incident_rows,
        f"SAR prediction error metrics by incident ({title_suffix})",
        dpi,
    )
    save_latex_table(
        output_dir / "incident_metrics_table.tex",
        incident_rows,
        f"SAR prediction error metrics by incident ({title_suffix})",
        f"tab:sar-incident-errors-{title_suffix.lower().replace(' ', '-')}",
    )


def scale_rows_for_incidents(
    *,
    incident_summaries,
    methods: list[str],
    scale_info,
) -> list[dict[str, str]]:
    grouped = defaultdict(list)
    for summary in incident_summaries:
        info = scale_info.get(summary.incident_index)
        if info is None:
            raise ValueError(
                f"Missing scale info for incident {summary.incident_index}"
            )
        grouped[(info.meters_per_cell, info.ground_width_m)].append(summary)

    rows: list[dict[str, str]] = []
    for meters_per_cell, ground_width_m in sorted(grouped):
        summaries = grouped[(meters_per_cell, ground_width_m)]
        row = {
            "meters_per_cell": format_scale(meters_per_cell),
            "ground_width_m": format_float(ground_width_m),
            "incident_count": str(len(summaries)),
        }
        for method in methods:
            stats = combine_stats(
                [
                    summary.methods.get(method, MetricStats(0, None, None))
                    for summary in summaries
                ]
            )
            row[f"{method}_prediction_count"] = str(stats.count)
            row[f"{method}_incident_count"] = str(
                sum(
                    1
                    for summary in summaries
                    if summary.methods.get(method, MetricStats(0, None, None)).count > 0
                )
            )
            row[f"{method}_avg_distance_m"] = fmt(stats.mean)
            row[f"{method}_stddev_distance_m"] = fmt(stats.stddev)
        rows.append(row)
    return rows


def write_scale_csv(path: Path, rows: list[dict[str, str]], methods: list[str]) -> None:
    fieldnames = ["meters_per_cell", "ground_width_m", "incident_count"]
    for method in methods:
        fieldnames.extend(
            [
                f"{method}_prediction_count",
                f"{method}_incident_count",
                f"{method}_avg_distance_m",
                f"{method}_stddev_distance_m",
            ]
        )

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_scale_outputs(
    *,
    output_dir: Path,
    incident_summaries,
    methods: list[str],
    scale_info,
    dpi: int,
) -> None:
    rows = scale_rows_for_incidents(
        incident_summaries=incident_summaries,
        methods=methods,
        scale_info=scale_info,
    )
    write_scale_csv(output_dir / "error_summary_by_map_scale.csv", rows, methods)
    if not rows:
        return

    summaries = []
    for row in rows:
        summaries.append(
            ScaleSummary(
                ground_width_m=float(row["ground_width_m"]),
                incident_count=int(row["incident_count"]),
                methods={
                    method: ScaleMethodStats(
                        prediction_count=int(row[f"{method}_prediction_count"]),
                        avg_distance_m=float(row[f"{method}_avg_distance_m"]),
                        stddev_distance_m=float(row[f"{method}_stddev_distance_m"]),
                    )
                    for method in methods
                    if row[f"{method}_avg_distance_m"]
                    and row[f"{method}_stddev_distance_m"]
                },
            )
        )

    summaries = [
        summary
        for summary in summaries
        if all(method in summary.methods for method in methods)
    ]
    if summaries:
        save_scale_plot(
            summaries=summaries,
            methods=methods,
            output_png=output_dir / "average_distance_by_map_scale.png",
            dpi=dpi,
        )


def main() -> None:
    args = parse_args()
    methods = infer_methods(args.combined_summary_csv)
    incidents_by_version_answer, answer_counts = load_answer_incidents(
        args.saliency_eval_parsed_json
    )
    all_summaries = merge_incident_summaries(
        read_summary_csv(args.combined_summary_csv, methods),
        methods,
    )
    summaries_by_incident = {
        summary.incident_index: summary for summary in all_summaries
    }
    scale_info = load_scale_info(args.ground_truth_csv)

    output_summary: dict[str, Any] = {
        "source": {
            "saliency_eval_parsed_json": str(args.saliency_eval_parsed_json),
            "combined_summary_csv": str(args.combined_summary_csv),
            "ground_truth_csv": str(args.ground_truth_csv),
        },
        "counts": {},
        "incident_lists": {},
    }

    split_specs = [
        ("salient", "yes"),
        ("non-salient", "no"),
    ]
    for split_name, answer in split_specs:
        for image_version in ("v1", "v2"):
            incident_indices = sorted(
                incidents_by_version_answer.get(image_version, {}).get(answer, set())
            )
            incident_summaries = [
                summaries_by_incident[incident_index]
                for incident_index in incident_indices
                if incident_index in summaries_by_incident
            ]
            output_dir = args.output_root / split_name / image_version
            title_suffix = f"{split_name}, {image_version}"
            write_summary_outputs_to_dir(
                output_dir=output_dir,
                incident_summaries=incident_summaries,
                methods=methods,
                dpi=args.dpi,
                title_suffix=title_suffix,
            )
            write_scale_outputs(
                output_dir=output_dir,
                incident_summaries=incident_summaries,
                methods=methods,
                scale_info=scale_info,
                dpi=args.dpi,
            )

            output_summary["counts"].setdefault(image_version, {})[answer] = len(
                incident_indices
            )
            output_summary["incident_lists"].setdefault(image_version, {})[
                answer
            ] = incident_indices
            print(
                f"{split_name}/{image_version}: "
                f"{len(incident_indices)} incidents -> {output_dir}"
            )

    output_summary["raw_answer_counts"] = {
        version: dict(counts) for version, counts in sorted(answer_counts.items())
    }
    summary_path = args.output_root / "saliency_eval_split_summary.json"
    summary_path.write_text(
        json.dumps(output_summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"summary_json: {summary_path}")


if __name__ == "__main__":
    main()
