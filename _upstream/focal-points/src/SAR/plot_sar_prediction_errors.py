import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

try:
    from .sar_method_colors import METHOD_COLORS
except ImportError:
    from sar_method_colors import METHOD_COLORS


DEFAULT_OUTPUT_DIR = Path("data/SAR_graphs")
DEFAULT_METHODS = ("vanilla", "saliency")


@dataclass(frozen=True)
class MetricStats:
    count: int
    mean: float | None
    stddev: float | None


@dataclass
class IncidentSummary:
    incident_index: int
    ground_truth_x_m: float | None
    ground_truth_y_m: float | None
    methods: dict[str, MetricStats]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create SAR prediction-error plots, PNG tables, and LaTeX tables from "
            "one or more summarize_sar_prediction_errors.py CSV outputs."
        )
    )
    parser.add_argument(
        "--summary-csv",
        nargs="+",
        type=Path,
        required=True,
        help="One or more SAR error summary CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Root output directory. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
        help='Methods to plot. Defaults to "vanilla saliency".',
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="PNG output DPI.",
    )
    return parser.parse_args()


def parse_optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def read_summary_csv(path: Path, methods: list[str]) -> list[IncidentSummary]:
    summaries: list[IncidentSummary] = []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"{path} is empty or missing a header")

        required = {"scope", "incident_index"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

        for row in reader:
            if row.get("scope") != "incident":
                continue
            incident_index_text = row.get("incident_index")
            if not incident_index_text:
                continue

            method_stats: dict[str, MetricStats] = {}
            for method in methods:
                count = int(row.get(f"{method}_prediction_count") or 0)
                method_stats[method] = MetricStats(
                    count=count,
                    mean=parse_optional_float(row.get(f"{method}_avg_distance_m")),
                    stddev=parse_optional_float(row.get(f"{method}_stddev_distance_m")),
                )

            summaries.append(
                IncidentSummary(
                    incident_index=int(incident_index_text),
                    ground_truth_x_m=parse_optional_float(row.get("ground_truth_x_m")),
                    ground_truth_y_m=parse_optional_float(row.get("ground_truth_y_m")),
                    methods=method_stats,
                )
            )

    return summaries


def combine_stats(stats: list[MetricStats]) -> MetricStats:
    stats = [stat for stat in stats if stat.count > 0 and stat.mean is not None]
    total_count = sum(stat.count for stat in stats)
    if total_count == 0:
        return MetricStats(count=0, mean=None, stddev=None)

    combined_mean = sum(stat.count * float(stat.mean) for stat in stats) / total_count
    spread_sum = 0.0
    for stat in stats:
        spread = float(stat.stddev or 0.0) ** 2
        mean_delta = float(stat.mean) - combined_mean
        spread_sum += stat.count * (spread + mean_delta**2)
    combined_spread = spread_sum / total_count
    return MetricStats(
        count=total_count,
        mean=combined_mean,
        stddev=math.sqrt(combined_spread),
    )


def merge_incident_summaries(
    summaries: list[IncidentSummary], methods: list[str]
) -> list[IncidentSummary]:
    grouped: dict[int, list[IncidentSummary]] = {}
    for summary in summaries:
        grouped.setdefault(summary.incident_index, []).append(summary)

    merged: list[IncidentSummary] = []
    for incident_index, incident_rows in sorted(grouped.items()):
        gt_x_values = {
            row.ground_truth_x_m
            for row in incident_rows
            if row.ground_truth_x_m is not None
        }
        gt_y_values = {
            row.ground_truth_y_m
            for row in incident_rows
            if row.ground_truth_y_m is not None
        }
        if len(gt_x_values) > 1 or len(gt_y_values) > 1:
            raise ValueError(f"Conflicting ground truth for incident {incident_index}")

        merged.append(
            IncidentSummary(
                incident_index=incident_index,
                ground_truth_x_m=next(iter(gt_x_values), None),
                ground_truth_y_m=next(iter(gt_y_values), None),
                methods={
                    method: combine_stats(
                        [
                            row.methods.get(method, MetricStats(0, None, None))
                            for row in incident_rows
                        ]
                    )
                    for method in methods
                },
            )
        )
    return merged


def overall_summary(
    incident_summaries: list[IncidentSummary], methods: list[str]
) -> IncidentSummary:
    return IncidentSummary(
        incident_index=0,
        ground_truth_x_m=None,
        ground_truth_y_m=None,
        methods={
            method: combine_stats(
                [
                    summary.methods.get(method, MetricStats(0, None, None))
                    for summary in incident_summaries
                ]
            )
            for method in methods
        },
    )


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.3f}"


def summary_row(
    scope: str, summary: IncidentSummary, methods: list[str]
) -> dict[str, str]:
    row = {
        "scope": scope,
        "incident_index": "" if scope == "overall" else str(summary.incident_index),
        "incident_count": "1" if scope == "incident" else "",
        "ground_truth_x_m": fmt(summary.ground_truth_x_m),
        "ground_truth_y_m": fmt(summary.ground_truth_y_m),
    }
    for method in methods:
        stats = summary.methods.get(method, MetricStats(0, None, None))
        row[f"{method}_prediction_count"] = str(stats.count)
        row[f"{method}_avg_distance_m"] = fmt(stats.mean)
        row[f"{method}_stddev_distance_m"] = fmt(stats.stddev)
    return row


def output_fieldnames(methods: list[str]) -> list[str]:
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


def write_combined_csv(
    path: Path,
    incident_summaries: list[IncidentSummary],
    overall: IncidentSummary,
    methods: list[str],
) -> None:
    rows = [summary_row("incident", summary, methods) for summary in incident_summaries]
    overall_row = summary_row("overall", overall, methods)
    overall_row["incident_count"] = str(len(incident_summaries))
    rows.append(overall_row)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=output_fieldnames(methods))
        writer.writeheader()
        writer.writerows(rows)


def color_for_method(method: str) -> str:
    return METHOD_COLORS.get(method, "#666666")


def label_for_method(method: str) -> str:
    return method.replace("_", " ").title()


def save_bar_chart(
    *,
    labels: list[str],
    values: list[float | None],
    title: str,
    ylabel: str,
    output_path: Path,
    dpi: int,
) -> None:
    filtered = [
        (label, value)
        for label, value in zip(labels, values)
        if value is not None and not math.isnan(value)
    ]
    if not filtered:
        return

    chart_labels = [item[0] for item in filtered]
    display_labels = [label_for_method(label) for label in chart_labels]
    chart_values = [float(item[1]) for item in filtered]
    colors = [color_for_method(label) for label in chart_labels]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    x_values = list(range(len(chart_labels)))
    ax.bar(x_values, chart_values, color=colors)
    ax.set_xticks(x_values)
    ax.set_xticklabels(display_labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.28)
    ax.set_axisbelow(True)
    for index, value in enumerate(chart_values):
        ax.text(index, value, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def save_line_chart(
    *,
    incident_summaries: list[IncidentSummary],
    methods: list[str],
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
    dpi: int,
) -> None:
    incidents = [summary.incident_index for summary in incident_summaries]
    if not incidents:
        return

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    plotted = False
    for method in methods:
        values: list[float | None] = []
        for summary in incident_summaries:
            stats = summary.methods.get(method, MetricStats(0, None, None))
            if metric == "count":
                values.append(float(stats.count))
            elif metric == "mean":
                values.append(stats.mean)
            elif metric == "stddev":
                values.append(stats.stddev)
            else:
                raise ValueError(f"Unsupported metric: {metric}")

        x_values = []
        y_values = []
        for incident, value in zip(incidents, values):
            if value is not None:
                x_values.append(incident)
                y_values.append(value)
        if not x_values:
            continue
        plotted = True
        ax.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=2.0,
            color=color_for_method(method),
            label=label_for_method(method),
        )

    if not plotted:
        plt.close(fig)
        return

    ax.set_title(title)
    ax.set_xlabel("Incident")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.28)
    ax.legend()
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def save_ground_truth_plot(
    *,
    incident_summaries: list[IncidentSummary],
    output_path: Path,
    dpi: int,
    title: str,
) -> None:
    points = [
        summary
        for summary in incident_summaries
        if summary.ground_truth_x_m is not None and summary.ground_truth_y_m is not None
    ]
    if not points:
        return

    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    x_values = [float(summary.ground_truth_x_m) for summary in points]
    y_values = [float(summary.ground_truth_y_m) for summary in points]
    ax.scatter(x_values, y_values, color="#2f6fbb", s=46)
    for summary, x_value, y_value in zip(points, x_values, y_values):
        ax.annotate(
            str(summary.incident_index),
            (x_value, y_value),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.axvline(0, color="#222222", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Ground truth x (m)")
    ax.set_ylabel("Ground truth y (m)")
    ax.grid(alpha=0.25)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def table_rows_for_summary(
    summary: IncidentSummary, methods: list[str]
) -> list[list[str]]:
    rows = [["metric", *[label_for_method(method) for method in methods]]]
    rows.append(
        [
            "prediction count",
            *[
                str(summary.methods.get(method, MetricStats(0, None, None)).count)
                for method in methods
            ],
        ]
    )
    rows.append(
        [
            "avg distance (m)",
            *[
                fmt(summary.methods.get(method, MetricStats(0, None, None)).mean)
                for method in methods
            ],
        ]
    )
    rows.append(
        [
            "stddev (m)",
            *[
                fmt(summary.methods.get(method, MetricStats(0, None, None)).stddev)
                for method in methods
            ],
        ]
    )
    if summary.ground_truth_x_m is not None and summary.ground_truth_y_m is not None:
        ground_truth_x = fmt(summary.ground_truth_x_m)
        ground_truth_y = fmt(summary.ground_truth_y_m)
        rows.append(["ground truth x (m)", ground_truth_x, *[""] * (len(methods) - 1)])
        rows.append(["ground truth y (m)", ground_truth_y, *[""] * (len(methods) - 1)])
    return rows


def table_rows_for_incidents(
    incident_summaries: list[IncidentSummary], methods: list[str]
) -> list[list[str]]:
    rows = [
        [
            "incident",
            "truth x",
            "truth y",
            *[
                label
                for method in methods
                for label in (
                    f"{method} n",
                    f"{method} avg",
                    f"{method} std",
                )
            ],
        ]
    ]
    for summary in incident_summaries:
        row = [
            str(summary.incident_index),
            fmt(summary.ground_truth_x_m),
            fmt(summary.ground_truth_y_m),
        ]
        for method in methods:
            stats = summary.methods.get(method, MetricStats(0, None, None))
            row.extend(
                [
                    str(stats.count),
                    fmt(stats.mean),
                    fmt(stats.stddev),
                ]
            )
        rows.append(row)
    return rows


def latex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def is_shared_ground_truth_row(row: list[str]) -> bool:
    return bool(row) and row[0] in {"ground truth x (m)", "ground truth y (m)"}


def save_latex_table(
    path: Path, rows: list[list[str]], caption: str, label: str
) -> None:
    column_spec = "l" + "r" * (len(rows[0]) - 1)
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{{latex_escape(caption)}}}",
        f"\\label{{{latex_escape(label)}}}",
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\hline",
        " & ".join(latex_escape(value) for value in rows[0]) + " \\\\",
        "\\hline",
    ]
    for row in rows[1:]:
        if is_shared_ground_truth_row(row) and len(row) > 2:
            value_colspan = len(rows[0]) - 1
            lines.append(
                f"{latex_escape(row[0])} & "
                f"\\multicolumn{{{value_colspan}}}{{c}}{{{latex_escape(row[1])}}} \\\\"
            )
        else:
            lines.append(" & ".join(latex_escape(value) for value in row) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def save_png_table(path: Path, rows: list[list[str]], title: str, dpi: int) -> None:
    col_count = len(rows[0])
    row_count = len(rows)
    fig_width = max(7.0, min(22.0, 1.25 * col_count))
    fig_height = max(2.4, min(28.0, 0.34 * row_count + 1.0))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(title, fontweight="bold", pad=10)

    col_units = []
    for col in range(col_count):
        max_len = max(len(str(row[col])) if col < len(row) else 0 for row in rows)
        col_units.append(max(1.0, min(3.2, max_len * 0.12)))
    total_units = sum(col_units)
    table_left = 0.04
    table_width = 0.92
    table_top = 0.86
    table_height = min(0.78, 0.12 * row_count)
    row_height = table_height / row_count

    x_positions = [table_left]
    for unit in col_units:
        x_positions.append(x_positions[-1] + table_width * unit / total_units)

    def draw_cell(
        *,
        row_index: int,
        start_col: int,
        end_col: int,
        text: str,
        header: bool = False,
    ) -> None:
        x = x_positions[start_col]
        y = table_top - (row_index + 1) * row_height
        width = x_positions[end_col + 1] - x
        ax.add_patch(
            Rectangle(
                (x, y),
                width,
                row_height,
                transform=ax.transAxes,
                facecolor="#f0f2f5" if header else "white",
                edgecolor="#d5d8de",
                linewidth=1.0,
            )
        )
        ax.text(
            x + width / 2,
            y + row_height / 2,
            text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold" if header else "normal",
        )

    for col, value in enumerate(rows[0]):
        draw_cell(row_index=0, start_col=col, end_col=col, text=value, header=True)

    for row_index, row_values in enumerate(rows[1:], start=1):
        if is_shared_ground_truth_row(row_values) and col_count > 2:
            draw_cell(
                row_index=row_index,
                start_col=0,
                end_col=0,
                text=row_values[0],
            )
            draw_cell(
                row_index=row_index,
                start_col=1,
                end_col=col_count - 1,
                text=row_values[1],
            )
            continue

        for col in range(col_count):
            draw_cell(
                row_index=row_index,
                start_col=col,
                end_col=col,
                text=row_values[col] if col < len(row_values) else "",
            )

    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def write_incident_outputs(
    summary: IncidentSummary, methods: list[str], output_dir: Path, dpi: int
) -> None:
    incident_dir = output_dir / f"incident_{summary.incident_index}"
    incident_dir.mkdir(parents=True, exist_ok=True)

    labels = methods
    save_bar_chart(
        labels=labels,
        values=[
            summary.methods.get(method, MetricStats(0, None, None)).mean
            for method in methods
        ],
        title=f"Incident {summary.incident_index}: average distance from ground truth",
        ylabel="Average distance (m)",
        output_path=incident_dir / "average_distance.png",
        dpi=dpi,
    )
    save_bar_chart(
        labels=labels,
        values=[
            summary.methods.get(method, MetricStats(0, None, None)).stddev
            for method in methods
        ],
        title=f"Incident {summary.incident_index}: distance standard deviation",
        ylabel="Standard deviation (m)",
        output_path=incident_dir / "stddev_distance.png",
        dpi=dpi,
    )
    table_rows = table_rows_for_summary(summary, methods)
    save_png_table(
        incident_dir / "metrics_table.png",
        table_rows,
        f"Incident {summary.incident_index} metrics",
        dpi,
    )
    save_latex_table(
        incident_dir / "metrics_table.tex",
        table_rows,
        f"Incident {summary.incident_index} SAR prediction error metrics",
        f"tab:sar-incident-{summary.incident_index}",
    )


def write_shared_outputs(
    incident_summaries: list[IncidentSummary],
    overall: IncidentSummary,
    methods: list[str],
    output_dir: Path,
    dpi: int,
) -> None:
    shared_dir = output_dir / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)

    write_combined_csv(
        shared_dir / "combined_error_summary.csv",
        incident_summaries,
        overall,
        methods,
    )
    save_line_chart(
        incident_summaries=incident_summaries,
        methods=methods,
        metric="mean",
        title="Average distance from ground truth by incident",
        ylabel="Average distance (m)",
        output_path=shared_dir / "average_distance_by_incident.png",
        dpi=dpi,
    )
    save_line_chart(
        incident_summaries=incident_summaries,
        methods=methods,
        metric="stddev",
        title="Distance standard deviation by incident",
        ylabel="Standard deviation (m)",
        output_path=shared_dir / "stddev_by_incident.png",
        dpi=dpi,
    )
    save_ground_truth_plot(
        incident_summaries=incident_summaries,
        output_path=shared_dir / "ground_truth_locations.png",
        dpi=dpi,
        title="Ground-truth find coordinates",
    )
    save_bar_chart(
        labels=methods,
        values=[
            overall.methods.get(method, MetricStats(0, None, None)).mean
            for method in methods
        ],
        title="Overall average distance from ground truth",
        ylabel="Average distance (m)",
        output_path=shared_dir / "overall_average_distance.png",
        dpi=dpi,
    )
    save_bar_chart(
        labels=methods,
        values=[
            overall.methods.get(method, MetricStats(0, None, None)).stddev
            for method in methods
        ],
        title="Overall distance standard deviation",
        ylabel="Standard deviation (m)",
        output_path=shared_dir / "overall_stddev.png",
        dpi=dpi,
    )

    overall_rows = table_rows_for_summary(overall, methods)
    save_png_table(
        shared_dir / "overall_table.png",
        overall_rows,
        "Overall SAR prediction error metrics",
        dpi,
    )
    save_latex_table(
        shared_dir / "overall_table.tex",
        overall_rows,
        "Overall SAR prediction error metrics",
        "tab:sar-overall-errors",
    )

    incident_rows = table_rows_for_incidents(incident_summaries, methods)
    save_png_table(
        shared_dir / "incident_metrics_table.png",
        incident_rows,
        "SAR prediction error metrics by incident",
        dpi,
    )
    save_latex_table(
        shared_dir / "incident_metrics_table.tex",
        incident_rows,
        "SAR prediction error metrics by incident",
        "tab:sar-incident-errors",
    )


def main() -> None:
    args = parse_args()
    methods = list(dict.fromkeys(args.methods))
    missing = [str(path) for path in args.summary_csv if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing summary CSV file(s):\n" + "\n".join(missing))

    incident_summaries = merge_incident_summaries(
        [
            summary
            for path in args.summary_csv
            for summary in read_summary_csv(path, methods)
        ],
        methods,
    )
    if not incident_summaries:
        raise ValueError("No incident rows found in the provided summary CSV files")

    overall = overall_summary(incident_summaries, methods)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for summary in incident_summaries:
        write_incident_outputs(summary, methods, args.output_dir, args.dpi)
    write_shared_outputs(
        incident_summaries, overall, methods, args.output_dir, args.dpi
    )

    print(f"incidents: {len(incident_summaries)}")
    print(f"output_dir: {args.output_dir}")
    print(f"shared_dir: {args.output_dir / 'shared'}")


if __name__ == "__main__":
    main()
