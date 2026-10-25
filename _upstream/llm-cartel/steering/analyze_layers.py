#!/usr/bin/env python3
"""
Analyze Layers - Layer-wise Analysis CLI

Provides detailed analysis and visualization of CAA performance across layers.
Can compare metrics from different training runs and identify optimal layers.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

# Add lib directory to path
sys.path.insert(0, str(Path(__file__).parent))

from lib import Visualizer


def load_metrics(metrics_path: str) -> Dict:
    """Load metrics from JSON file"""
    with open(metrics_path, 'r') as f:
        data = json.load(f)
    return data


def print_layer_summary(metrics_data: Dict, top_n: int = 10):
    """Print detailed summary of layer performance"""
    layer_metrics = {
        int(k): v for k, v in metrics_data['layer_metrics'].items()
    }

    layers = sorted(layer_metrics.keys())

    print("\n" + "="*80)
    print("LAYER-WISE PERFORMANCE SUMMARY")
    print("="*80)

    # Overall statistics
    cohen_ds = [m['cohen_d'] for m in layer_metrics.values()]
    accuracies = [m['accuracy'] for m in layer_metrics.values()]
    fisher_ratios = [m['fisher_ratio'] for m in layer_metrics.values()]

    print(f"\nTotal layers analyzed: {len(layers)}")
    print(f"\nCohen's d statistics:")
    print(f"  Mean: {sum(cohen_ds)/len(cohen_ds):.3f}")
    print(f"  Std: {(sum((x - sum(cohen_ds)/len(cohen_ds))**2 for x in cohen_ds) / len(cohen_ds))**0.5:.3f}")
    print(f"  Min: {min(cohen_ds):.3f} (layer {layers[cohen_ds.index(min(cohen_ds))]})")
    print(f"  Max: {max(cohen_ds):.3f} (layer {layers[cohen_ds.index(max(cohen_ds))]})")

    print(f"\nAccuracy statistics:")
    print(f"  Mean: {sum(accuracies)/len(accuracies):.3f}")
    print(f"  Min: {min(accuracies):.3f}")
    print(f"  Max: {max(accuracies):.3f}")

    # Top layers by different metrics
    sorted_by_cohen = sorted(layer_metrics.items(), key=lambda x: x[1]['cohen_d'], reverse=True)
    sorted_by_accuracy = sorted(layer_metrics.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    sorted_by_fisher = sorted(layer_metrics.items(), key=lambda x: x[1]['fisher_ratio'], reverse=True)

    print(f"\nTop {top_n} layers by Cohen's d:")
    print(f"{'Layer':<8} {'Cohen\'s d':<12} {'Accuracy':<12} {'Fisher':<12} {'AUC-ROC':<12} {'p-value':<12}")
    print("-" * 80)
    for layer, m in sorted_by_cohen[:top_n]:
        print(f"{layer:<8} {m['cohen_d']:<12.3f} {m['accuracy']:<12.3f} "
              f"{m['fisher_ratio']:<12.3f} {m['auc_roc']:<12.3f} {m['p_value']:<12.2e}")

    print(f"\nTop {top_n} layers by accuracy:")
    print(f"{'Layer':<8} {'Accuracy':<12} {'Cohen\'s d':<12} {'Overlap':<12} {'Min Margin':<12}")
    print("-" * 80)
    for layer, m in sorted_by_accuracy[:top_n]:
        print(f"{layer:<8} {m['accuracy']:<12.3f} {m['cohen_d']:<12.3f} "
              f"{m['overlap_percent']:<12.3f} {m['min_margin']:<12.3f}")

    print(f"\nTop {top_n} layers by Fisher ratio (SNR):")
    print(f"{'Layer':<8} {'Fisher':<12} {'Cohen\'s d':<12} {'Mean Sep':<12} {'t-stat':<12}")
    print("-" * 80)
    for layer, m in sorted_by_fisher[:top_n]:
        print(f"{layer:<8} {m['fisher_ratio']:<12.3f} {m['cohen_d']:<12.3f} "
              f"{m['separation']:<12.3f} {m['t_statistic']:<12.3f}")

    # Layer range analysis
    print("\n" + "="*80)
    print("LAYER RANGE ANALYSIS")
    print("="*80)

    num_layers = len(layers)
    early_layers = [l for l in layers if l < num_layers // 3]
    middle_layers = [l for l in layers if num_layers // 3 <= l < 2 * num_layers // 3]
    late_layers = [l for l in layers if l >= 2 * num_layers // 3]

    for range_name, range_layers in [("Early", early_layers), ("Middle", middle_layers), ("Late", late_layers)]:
        if range_layers:
            range_cohen = [layer_metrics[l]['cohen_d'] for l in range_layers]
            range_accuracy = [layer_metrics[l]['accuracy'] for l in range_layers]
            print(f"\n{range_name} layers ({min(range_layers)}-{max(range_layers)}):")
            print(f"  Mean Cohen's d: {sum(range_cohen)/len(range_cohen):.3f}")
            print(f"  Mean Accuracy: {sum(range_accuracy)/len(range_accuracy):.3f}")
            print(f"  Best layer: {max(range_layers, key=lambda l: layer_metrics[l]['cohen_d'])} "
                  f"(Cohen's d = {layer_metrics[max(range_layers, key=lambda l: layer_metrics[l]['cohen_d'])]['cohen_d']:.3f})")

    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    best_layer = sorted_by_cohen[0][0]
    best_metrics = sorted_by_cohen[0][1]

    print(f"\nBest overall layer: {best_layer}")
    print(f"  Cohen's d: {best_metrics['cohen_d']:.3f} (effect size: {'huge' if best_metrics['cohen_d'] > 2 else 'large' if best_metrics['cohen_d'] > 0.8 else 'medium'})")
    print(f"  Accuracy: {best_metrics['accuracy']:.3f}")
    print(f"  AUC-ROC: {best_metrics['auc_roc']:.3f}")
    print(f"  Statistical significance: p = {best_metrics['p_value']:.2e}")

    print(f"\nTop 3 layers for ensemble:")
    for i, (layer, m) in enumerate(sorted_by_cohen[:3], 1):
        print(f"  {i}. Layer {layer} (Cohen's d = {m['cohen_d']:.3f}, Accuracy = {m['accuracy']:.3f})")

    # Quality assessment
    print(f"\nQuality assessment:")
    if best_metrics['cohen_d'] > 2.0:
        quality = "EXCELLENT - Very strong separation"
    elif best_metrics['cohen_d'] > 0.8:
        quality = "GOOD - Large effect size"
    elif best_metrics['cohen_d'] > 0.5:
        quality = "MODERATE - Medium effect size"
    else:
        quality = "WEAK - Small effect size, may not be reliable"
    print(f"  {quality}")

    if best_metrics['p_value'] < 0.001:
        print(f"  Highly statistically significant (p < 0.001)")
    elif best_metrics['p_value'] < 0.05:
        print(f"  Statistically significant (p < 0.05)")
    else:
        print(f"  WARNING: Not statistically significant (p = {best_metrics['p_value']:.3f})")


def compare_multiple_runs(metrics_files: List[str], names: List[str]):
    """Compare metrics from multiple training runs"""
    print("\n" + "="*80)
    print("COMPARING MULTIPLE TRAINING RUNS")
    print("="*80)

    all_metrics = []
    for path, name in zip(metrics_files, names):
        data = load_metrics(path)
        all_metrics.append((name, data))
        print(f"\n{name}:")
        print(f"  File: {Path(path).name}")
        print(f"  Model: {data.get('model_name', 'unknown')}")
        print(f"  Best layer: {data.get('best_layer', 'unknown')}")
        print(f"  Num layers: {data.get('num_layers', 'unknown')}")

    # Compare best layers
    print("\n" + "-"*80)
    print("Best layer comparison:")
    print(f"{'Run':<20} {'Best Layer':<15} {'Cohen\'s d':<15} {'Accuracy':<15}")
    print("-"*80)

    for name, data in all_metrics:
        best_layer = str(data.get('best_layer', 0))
        if best_layer in data['layer_metrics']:
            m = data['layer_metrics'][best_layer]
            print(f"{name:<20} {best_layer:<15} {m['cohen_d']:<15.3f} {m['accuracy']:<15.3f}")

    # Find common strong layers
    print("\nCommon strong layers (Cohen's d > 1.0):")
    layer_sets = []
    for name, data in all_metrics:
        strong_layers = {
            int(l) for l, m in data['layer_metrics'].items()
            if m['cohen_d'] > 1.0
        }
        layer_sets.append(strong_layers)
        print(f"  {name}: {sorted(strong_layers)}")

    if len(layer_sets) > 1:
        common_layers = set.intersection(*layer_sets)
        if common_layers:
            print(f"\n  Common across all runs: {sorted(common_layers)}")
        else:
            print(f"\n  No layers are consistently strong across all runs")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze CAA performance across layers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single training run:
  python analyze_layers.py --metrics results/steering_vectors/max_profits/metrics.json

  # Compare multiple runs:
  python analyze_layers.py --compare \\
    --metrics results/steering_vectors/max_profits/metrics.json \\
             results/steering_vectors/explicit_collusion/metrics.json \\
    --names "Max Profits" "Explicit Collusion"

  # Create visualizations from metrics:
  python analyze_layers.py --metrics results/steering_vectors/max_profits/metrics.json \\
    --visualize --output plots/
        """
    )

    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to metrics JSON file(s)"
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple metrics files"
    )

    parser.add_argument(
        "--names",
        type=str,
        nargs="+",
        help="Names for each metrics file (for comparison mode)"
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top layers to display (default: 10)"
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for visualizations (default: same as metrics)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.compare and len(args.metrics) < 2:
        parser.error("--compare requires at least 2 metrics files")

    if args.compare:
        # Use provided names or generate default ones
        if args.names:
            if len(args.names) != len(args.metrics):
                parser.error("Number of --names must match number of --metrics")
            names = args.names
        else:
            names = [f"Run {i+1}" for i in range(len(args.metrics))]

        # Compare multiple runs
        compare_multiple_runs(args.metrics, names)

    else:
        # Analyze single run
        metrics_path = args.metrics[0]

        if not Path(metrics_path).exists():
            print(f"Error: Metrics file not found: {metrics_path}")
            sys.exit(1)

        print(f"\nLoading metrics from {metrics_path}")
        metrics_data = load_metrics(metrics_path)

        # Print summary
        print_layer_summary(metrics_data, top_n=args.top_n)

        # Generate visualizations if requested
        if args.visualize:
            # Determine output directory
            if args.output:
                output_dir = Path(args.output)
            else:
                output_dir = Path(metrics_path).parent / "plots"

            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n" + "="*80)
            print("GENERATING VISUALIZATIONS")
            print("="*80)
            print(f"\nOutput directory: {output_dir}")

            # Extract layer metrics
            layer_metrics = {
                int(k): v for k, v in metrics_data['layer_metrics'].items()
            }

            # Create visualizer
            visualizer = Visualizer(output_dir=str(output_dir))

            # Create layer analysis plot
            print("  Creating layer analysis plot...")
            model_name = metrics_data.get('model_name', 'Model').split('/')[-1]
            visualizer.plot_layer_analysis(
                layer_metrics,
                model_name=model_name,
                save_name="layer_analysis.png"
            )

            print(f"\nVisualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
