import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .sar_method_colors import METHOD_COLORS
except ImportError:
    from sar_method_colors import METHOD_COLORS

DEFAULT_OUTPUT_PNG = Path(
    "data/SAR_graphs/openai/gpt-5.5/q1_q2-q30_part1/shared/"
    "average_distance_by_map_scale.png"
)
DEFAULT_METHODS = ("vanilla", "saliency")


@dataclass(frozen=True)
class ScaleMethodStats:
    prediction_count: int
    avg_distance_m: float
    stddev_distance_m: float


@dataclass(frozen=True)
class ScaleSummary:
    ground_width_m: float
    incident_count: int
    methods: dict[str, ScaleMethodStats]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot SAR prediction error by map scale from a "
            "q*_error_summary_by_scale.csv file."
        )
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        required=True,
        help="By-scale summary CSV with avg/stddev distance columns.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=DEFAULT_OUTPUT_PNG,
        help=f"PNG output path. Defaults to {DEFAULT_OUTPUT_PNG}.",
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
        default=220,
        help="PNG output DPI.",
    )
    return parser.parse_args()


def parse_float(row: dict[str, str], field: str) -> float:
    value = row.get(field)
    if value is None or value == "":
        raise ValueError(f"Missing required numeric field: {field}")
    return float(value)


def parse_int(row: dict[str, str], field: str) -> int:
    value = row.get(field)
    if value is None or value == "":
        raise ValueError(f"Missing required integer field: {field}")
    return int(value)


def read_scale_summaries(path: Path, methods: list[str]) -> list[ScaleSummary]:
    summaries: list[ScaleSummary] = []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"{path} is empty or missing a header")

        required = {"ground_width_m", "incident_count"}
        for method in methods:
            required.update(
                {
                    f"{method}_prediction_count",
                    f"{method}_avg_distance_m",
                    f"{method}_stddev_distance_m",
                }
            )
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

        for row in reader:
            summaries.append(
                ScaleSummary(
                    ground_width_m=parse_float(row, "ground_width_m"),
                    incident_count=parse_int(row, "incident_count"),
                    methods={
                        method: ScaleMethodStats(
                            prediction_count=parse_int(
                                row,
                                f"{method}_prediction_count",
                            ),
                            avg_distance_m=parse_float(
                                row,
                                f"{method}_avg_distance_m",
                            ),
                            stddev_distance_m=parse_float(
                                row,
                                f"{method}_stddev_distance_m",
                            ),
                        )
                        for method in methods
                    },
                )
            )

    if not summaries:
        raise ValueError(f"No scale rows found in {path}")
    return sorted(summaries, key=lambda summary: summary.ground_width_m)


def color_for_method(method: str) -> str:
    return METHOD_COLORS.get(method, "#666666")


def label_for_method(method: str) -> str:
    return method.replace("_", " ").title()


def annotate_bar_values(ax, bars, counts: list[int]) -> None:
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 90,
            f"{height / 1000:.1f} km\nruns={count}",
            ha="center",
            va="bottom",
            fontsize=7,
        )


def save_scale_plot(
    *,
    summaries: list[ScaleSummary],
    methods: list[str],
    output_png: Path,
    dpi: int,
) -> None:
    labels = [f"{int(summary.ground_width_m / 1000)} km" for summary in summaries]
    incident_counts = [summary.incident_count for summary in summaries]
    x_values = list(range(len(summaries)))
    group_width = 0.78
    bar_width = min(0.28, group_width / len(methods))
    method_offsets = [
        (index - (len(methods) - 1) / 2) * bar_width for index in range(len(methods))
    ]

    fig_width = max(10.5, 8.0 + 0.7 * len(methods))
    fig, ax = plt.subplots(figsize=(fig_width, 6.8))
    max_error_top = 0.0
    plotted_bars = []

    for method, offset in zip(methods, method_offsets):
        averages = [summary.methods[method].avg_distance_m for summary in summaries]
        stddevs = [summary.methods[method].stddev_distance_m for summary in summaries]
        counts = [summary.methods[method].prediction_count for summary in summaries]
        positions = [x_value + offset for x_value in x_values]
        max_error_top = max(
            max_error_top,
            max(avg + stddev for avg, stddev in zip(averages, stddevs)),
        )
        bars = ax.bar(
            positions,
            averages,
            bar_width,
            yerr=stddevs,
            capsize=5,
            label=label_for_method(method),
            color=color_for_method(method),
            alpha=0.9,
        )
        plotted_bars.append((bars, counts))

    for index, summary in enumerate(summaries):
        best_method = min(
            methods,
            key=lambda method: summary.methods[method].avg_distance_m,
        )
        best_color = color_for_method(best_method)
        top = max(
            summary.methods[method].avg_distance_m
            + summary.methods[method].stddev_distance_m
            for method in methods
        )
        ax.text(
            index,
            top + 350,
            f"{label_for_method(best_method)} better",
            ha="center",
            va="bottom",
            fontsize=9,
            color=best_color,
            fontweight="bold",
        )

    for bars, counts in plotted_bars:
        annotate_bar_values(ax, bars, counts)

    ax.set_title("SAR Prediction Error by Map Scale", fontsize=16, pad=14)
    ax.set_xlabel("Map width / scale group", fontsize=12)
    ax.set_ylabel("Average distance to ground truth (meters)", fontsize=12)
    ax.set_xticks(x_values)
    ax.set_xticklabels(
        [
            f"{label}\nincidents={incident_count}"
            for label, incident_count in zip(labels, incident_counts)
        ]
    )
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=11)
    ax.set_ylim(0, max_error_top + 1400)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_png, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    methods = list(dict.fromkeys(args.methods))
    summaries = read_scale_summaries(args.summary_csv, methods)
    save_scale_plot(
        summaries=summaries,
        methods=methods,
        output_png=args.output_png,
        dpi=args.dpi,
    )
    print(f"output_png: {args.output_png}")


if __name__ == "__main__":
    main()
