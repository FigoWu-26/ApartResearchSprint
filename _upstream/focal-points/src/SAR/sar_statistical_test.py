import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

DEFAULT_SUMMARY_ROOT = Path("data/SAR_graphs/openai/gpt-5.5/q1-q65")
DEFAULT_GROUND_TRUTH_CSV = Path("data/SAR_prompts/find_location_ground_truth.csv")
DEFAULT_BASELINE_METHOD = "vanilla_v1"
DEFAULT_COMPARISON_METHODS = (
    "saliency_v1",
    "saliency_v2",
    "saliency_v3",
    "saliency_v4",
)
DEFAULT_SPLITS = (
    "shared",
    "salient/v2",
    "non-salient/v2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute paired significance tests for SAR method improvements over "
            "a baseline method, using incidents as the statistical unit."
        )
    )
    parser.add_argument(
        "--summary-root",
        type=Path,
        default=DEFAULT_SUMMARY_ROOT,
        help=f"Root graph folder containing split combined_error_summary.csv files. Defaults to {DEFAULT_SUMMARY_ROOT}.",
    )
    parser.add_argument(
        "--ground-truth-csv",
        type=Path,
        default=DEFAULT_GROUND_TRUTH_CSV,
        help=f"Ground-truth CSV with ground_width_m. Defaults to {DEFAULT_GROUND_TRUTH_CSV}.",
    )
    parser.add_argument(
        "--baseline-method",
        default=DEFAULT_BASELINE_METHOD,
        help=f"Baseline method column prefix. Defaults to {DEFAULT_BASELINE_METHOD}.",
    )
    parser.add_argument(
        "--comparison-methods",
        nargs="+",
        default=list(DEFAULT_COMPARISON_METHODS),
        help="Method column prefixes to compare with the baseline.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help=(
            "Split folders relative to --summary-root. Each must contain "
            "combined_error_summary.csv."
        ),
    )
    parser.add_argument(
        "--permutation-samples",
        type=int,
        default=100_000,
        help="Monte Carlo sign-flip permutation samples when exact enumeration is too large.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=30_000,
        help="Bootstrap samples for the mean improvement confidence interval.",
    )
    parser.add_argument(
        "--exact-max-n",
        type=int,
        default=20,
        help="Use exact sign-flip permutation test up to this number of paired incidents.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed for Monte Carlo permutation and bootstrap tests.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <summary-root>/method_significance_tests.csv.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to <summary-root>/method_significance_tests.json.",
    )
    return parser.parse_args()


def parse_float(value: str | None, field: str) -> float:
    if value is None or value == "":
        raise ValueError(f"Missing numeric value for {field}")
    return float(value)


def load_ground_widths(path: Path) -> dict[int, float]:
    widths: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"incident_index", "ground_width_m"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

        for row in reader:
            incident_index = int(row["incident_index"])
            widths[incident_index] = parse_float(
                row["ground_width_m"], "ground_width_m"
            )

    if not widths:
        raise ValueError(f"No ground-width rows found in {path}")
    return widths


def load_incident_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if row.get("scope") == "incident"]
    if not rows:
        raise ValueError(f"No incident rows found in {path}")
    return rows


def binomial_sign_test_p_value(positive_count: int, nonzero_count: int) -> float:
    if nonzero_count == 0:
        return float("nan")
    tail_sum = sum(
        math.comb(nonzero_count, count)
        for count in range(positive_count, nonzero_count + 1)
    )
    return tail_sum / (2**nonzero_count)


def permutation_p_value(
    values: list[float],
    *,
    samples: int,
    exact_max_n: int,
    rng: random.Random,
) -> float:
    if not values:
        return float("nan")

    observed = statistics.mean(values)
    n = len(values)
    if n <= exact_max_n:
        total = 2**n
        extreme = 0
        for mask in range(total):
            signed_sum = 0.0
            for index, value in enumerate(values):
                signed_sum += value if mask & (1 << index) else -value
            if signed_sum / n >= observed:
                extreme += 1
        return extreme / total

    extreme = 0
    for _ in range(samples):
        signed_mean = (
            sum(value if rng.random() < 0.5 else -value for value in values) / n
        )
        if signed_mean >= observed:
            extreme += 1
    return (extreme + 1) / (samples + 1)


def bootstrap_mean_ci(
    values: list[float],
    *,
    samples: int,
    rng: random.Random,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")

    n = len(values)
    means = [
        sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples)
    ]
    means.sort()
    low_index = max(0, min(samples - 1, int((alpha / 2) * samples)))
    high_index = max(0, min(samples - 1, int((1 - alpha / 2) * samples)))
    return means[low_index], means[high_index]


def safe_mean(values: list[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def safe_median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def significance_for_method(
    *,
    split_name: str,
    rows: list[dict[str, str]],
    ground_widths: dict[int, float],
    baseline_method: str,
    comparison_method: str,
    permutation_samples: int,
    bootstrap_samples: int,
    exact_max_n: int,
    seed: int,
) -> dict[str, Any]:
    raw_improvements: list[float] = []
    normalized_improvements: list[float] = []
    percent_reductions: list[float] = []
    incident_indices: list[int] = []

    for row in rows:
        incident_index = int(row["incident_index"])
        ground_width_m = ground_widths.get(incident_index)
        if ground_width_m is None:
            raise ValueError(f"Missing ground_width_m for incident {incident_index}")
        if ground_width_m <= 0:
            raise ValueError(
                f"ground_width_m must be positive for incident {incident_index}"
            )

        baseline_error = parse_float(
            row.get(f"{baseline_method}_avg_distance_m"),
            f"{baseline_method}_avg_distance_m",
        )
        comparison_error = parse_float(
            row.get(f"{comparison_method}_avg_distance_m"),
            f"{comparison_method}_avg_distance_m",
        )
        improvement = baseline_error - comparison_error

        incident_indices.append(incident_index)
        raw_improvements.append(improvement)
        normalized_improvements.append(improvement / ground_width_m)
        if baseline_error:
            percent_reductions.append(improvement / baseline_error)

    raw_nonzero = [value for value in raw_improvements if value != 0]
    raw_positive = sum(value > 0 for value in raw_improvements)
    raw_negative = sum(value < 0 for value in raw_improvements)
    raw_zero = sum(value == 0 for value in raw_improvements)
    raw_positive_nonzero = sum(value > 0 for value in raw_nonzero)

    raw_rng = random.Random(seed)
    normalized_rng = random.Random(seed + 1)
    raw_bootstrap_rng = random.Random(seed + 2)
    normalized_bootstrap_rng = random.Random(seed + 3)

    raw_ci_low, raw_ci_high = bootstrap_mean_ci(
        raw_improvements,
        samples=bootstrap_samples,
        rng=raw_bootstrap_rng,
    )
    normalized_ci_low, normalized_ci_high = bootstrap_mean_ci(
        normalized_improvements,
        samples=bootstrap_samples,
        rng=normalized_bootstrap_rng,
    )

    return {
        "split": split_name,
        "baseline_method": baseline_method,
        "comparison_method": comparison_method,
        "incident_count": len(raw_improvements),
        "incident_indices": incident_indices,
        "better_incident_count": raw_positive,
        "worse_incident_count": raw_negative,
        "tie_incident_count": raw_zero,
        "mean_improvement_m": safe_mean(raw_improvements),
        "median_improvement_m": safe_median(raw_improvements),
        "bootstrap_ci_low_m": raw_ci_low,
        "bootstrap_ci_high_m": raw_ci_high,
        "permutation_p_one_sided": permutation_p_value(
            raw_improvements,
            samples=permutation_samples,
            exact_max_n=exact_max_n,
            rng=raw_rng,
        ),
        "sign_test_p_one_sided": binomial_sign_test_p_value(
            raw_positive_nonzero,
            len(raw_nonzero),
        ),
        "mean_improvement_fraction_map_width": safe_mean(normalized_improvements),
        "median_improvement_fraction_map_width": safe_median(normalized_improvements),
        "mean_improvement_pct_map_width": safe_mean(normalized_improvements) * 100,
        "median_improvement_pct_map_width": safe_median(normalized_improvements) * 100,
        "bootstrap_ci_low_pct_map_width": normalized_ci_low * 100,
        "bootstrap_ci_high_pct_map_width": normalized_ci_high * 100,
        "normalized_permutation_p_one_sided": permutation_p_value(
            normalized_improvements,
            samples=permutation_samples,
            exact_max_n=exact_max_n,
            rng=normalized_rng,
        ),
        "mean_percent_reduction_vs_baseline": safe_mean(percent_reductions) * 100,
        "median_percent_reduction_vs_baseline": safe_median(percent_reductions) * 100,
    }


def output_fieldnames() -> list[str]:
    return [
        "split",
        "baseline_method",
        "comparison_method",
        "incident_count",
        "better_incident_count",
        "worse_incident_count",
        "tie_incident_count",
        "mean_improvement_m",
        "median_improvement_m",
        "bootstrap_ci_low_m",
        "bootstrap_ci_high_m",
        "permutation_p_one_sided",
        "sign_test_p_one_sided",
        "mean_improvement_pct_map_width",
        "median_improvement_pct_map_width",
        "bootstrap_ci_low_pct_map_width",
        "bootstrap_ci_high_pct_map_width",
        "normalized_permutation_p_one_sided",
        "mean_percent_reduction_vs_baseline",
        "median_percent_reduction_vs_baseline",
    ]


def format_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6f}"
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=output_fieldnames(), extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: format_value(row.get(key, "")) for key in output_fieldnames()}
            )


def main() -> None:
    args = parse_args()
    output_csv = args.output_csv or args.summary_root / "method_significance_tests.csv"
    output_json = (
        args.output_json or args.summary_root / "method_significance_tests.json"
    )

    ground_widths = load_ground_widths(args.ground_truth_csv)
    result_rows: list[dict[str, Any]] = []

    for split in args.splits:
        summary_csv = args.summary_root / split / "combined_error_summary.csv"
        if not summary_csv.exists():
            raise FileNotFoundError(f"Missing split summary CSV: {summary_csv}")

        incident_rows = load_incident_rows(summary_csv)
        for comparison_method in args.comparison_methods:
            result_rows.append(
                significance_for_method(
                    split_name=split,
                    rows=incident_rows,
                    ground_widths=ground_widths,
                    baseline_method=args.baseline_method,
                    comparison_method=comparison_method,
                    permutation_samples=args.permutation_samples,
                    bootstrap_samples=args.bootstrap_samples,
                    exact_max_n=args.exact_max_n,
                    seed=args.seed,
                )
            )

    write_csv(output_csv, result_rows)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "source": {
                    "summary_root": str(args.summary_root),
                    "ground_truth_csv": str(args.ground_truth_csv),
                },
                "test_definition": {
                    "improvement": "baseline avg distance - comparison avg distance",
                    "positive_improvement_means": "comparison method is closer to ground truth",
                    "statistical_unit": "incident",
                    "normalized_improvement": "improvement_m / ground_width_m",
                    "p_values": "one-sided tests for mean or sign improvement > 0",
                    "baseline_method": args.baseline_method,
                    "comparison_methods": args.comparison_methods,
                    "splits": args.splits,
                    "permutation_samples": args.permutation_samples,
                    "bootstrap_samples": args.bootstrap_samples,
                    "seed": args.seed,
                },
                "results": result_rows,
            },
            file,
            indent=2,
        )

    print(f"rows: {len(result_rows)}")
    print(f"output_csv: {output_csv}")
    print(f"output_json: {output_json}")


if __name__ == "__main__":
    main()
