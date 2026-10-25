#!/usr/bin/env python3
"""
Train CAA Steering Vectors - Main CLI

Unified pipeline for training Contrastive Activation Addition (CAA) steering vectors.
Extracts activations from all layers, computes vectors, and generates comprehensive visualizations.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add lib directory to path
sys.path.insert(0, str(Path(__file__).parent))

from lib import DataLoader, ActivationExtractor, CAATrainer, Visualizer


def list_datasets(datasets_dir: str = "results/datasets"):
    """List all available datasets"""
    print("\n" + "="*70)
    print("AVAILABLE DATASETS")
    print("="*70)

    datasets = DataLoader.list_available_datasets(datasets_dir)

    if not datasets:
        print(f"\nNo datasets found in {datasets_dir}")
        print("Datasets should be named: <name>_high_scores.jsonl and <name>_low_scores.jsonl")
        return

    print(f"\nFound {len(datasets)} complete dataset pairs:\n")
    for name, paths in sorted(datasets.items()):
        print(f"  {name}")
        print(f"    High: {Path(paths['high']).name}")
        print(f"    Low:  {Path(paths['low']).name}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Train CAA steering vectors for collusion detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available datasets:
  python train_steering_vectors.py --list-datasets

  # Train with default settings:
  python train_steering_vectors.py --dataset max_profits

  # Train with custom settings:
  python train_steering_vectors.py \\
    --dataset max_profits \\
    --max-pairs 100 \\
    --output-name experiment1 \\
    --no-visualize

  # Train with custom dataset files:
  python train_steering_vectors.py \\
    --high-data path/to/high.jsonl \\
    --low-data path/to/low.jsonl \\
    --output-name custom_experiment
        """
    )

    # Dataset selection
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument(
        "--dataset",
        type=str,
        help="Dataset name (without _high_scores/_low_scores suffix)"
    )
    dataset_group.add_argument(
        "--list-datasets",
        action="store_true",
        help="List all available datasets and exit"
    )

    # Custom dataset paths
    parser.add_argument(
        "--high-data",
        type=str,
        help="Path to high collusion JSONL file (overrides --dataset)"
    )
    parser.add_argument(
        "--low-data",
        type=str,
        help="Path to low collusion JSONL file (overrides --dataset)"
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        help="Model to extract activations from"
    )

    # Training configuration
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Maximum number of prompt-thought pairs to use (default: all)"
    )

    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Specific layers to extract (default: all layers)"
    )

    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Don't normalize vectors to unit length"
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/steering_vectors",
        help="Base directory for output files"
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Name for this training run (default: dataset name)"
    )

    # Activation caching
    parser.add_argument(
        "--load-activations",
        type=str,
        default=None,
        help="Path to pre-computed activations (skips extraction)"
    )

    parser.add_argument(
        "--skip-save-activations",
        action="store_true",
        help="Don't save intermediate activations"
    )

    # Visualization
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=True,
        help="Generate visualization plots (default: True)"
    )

    parser.add_argument(
        "--no-visualize",
        dest="visualize",
        action="store_false",
        help="Skip visualization generation"
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top layers to visualize in detail (default: 5)"
    )

    args = parser.parse_args()

    # Handle --list-datasets
    if args.list_datasets:
        list_datasets()
        return

    # Validate arguments
    if not args.dataset and not (args.high_data and args.low_data):
        parser.error("Must specify either --dataset or both --high-data and --low-data")

    if args.high_data and not args.low_data:
        parser.error("Must specify both --high-data and --low-data")

    if args.low_data and not args.high_data:
        parser.error("Must specify both --high-data and --low-data")

    # Determine dataset paths
    if args.high_data and args.low_data:
        high_data_path = args.high_data
        low_data_path = args.low_data
        dataset_name = args.output_name or "custom"
    else:
        # Use --dataset
        datasets = DataLoader.list_available_datasets()
        if args.dataset not in datasets:
            print(f"Error: Dataset '{args.dataset}' not found.")
            print("\nAvailable datasets:")
            for name in sorted(datasets.keys()):
                print(f"  - {name}")
            sys.exit(1)

        high_data_path = datasets[args.dataset]['high']
        low_data_path = datasets[args.dataset]['low']
        dataset_name = args.output_name or args.dataset

    # Create output directory structure
    output_dir = Path(args.output_dir) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Print configuration
    print("\n" + "="*70)
    print("CAA STEERING VECTOR TRAINING")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Dataset name: {dataset_name}")
    print(f"  High data: {Path(high_data_path).name}")
    print(f"  Low data: {Path(low_data_path).name}")
    print(f"  Model: {args.model}")
    print(f"  Max pairs: {args.max_pairs or 'all'}")
    print(f"  Layers: {args.layers or 'all'}")
    print(f"  Normalize: {not args.no_normalize}")
    print(f"  Output directory: {output_dir}")
    print(f"  Visualize: {args.visualize}")
    print("="*70)

    # Step 1: Load data
    print("\n[Step 1/5] Loading data...")
    loader = DataLoader(high_data_path, low_data_path)
    loader.load_data(verbose=True)

    # Get statistics
    stats = loader.get_statistics()
    print("\nDataset statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    # Get prompt-thought pairs
    prompt_thought_pairs = loader.get_prompt_thought_pairs(max_pairs=args.max_pairs)

    if len(prompt_thought_pairs) == 0:
        print("\nError: No valid prompt-thought pairs found")
        sys.exit(1)

    # Step 2: Extract activations or load pre-computed
    if args.load_activations:
        print(f"\n[Step 2/5] Loading pre-computed activations from {args.load_activations}")
        trainer = CAATrainer(args.model)
        layer_activations, metadata = trainer.load_activations(args.load_activations)
        print(f"Loaded {metadata['num_layers']} layers with {metadata['num_pairs']} pairs")

    else:
        print("\n[Step 2/5] Extracting activations...")
        print("This may take several minutes for large models and many pairs...")

        extractor = ActivationExtractor(args.model)

        # Get model info
        model_info = extractor.get_model_info()
        print(f"\nModel info:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")

        # Extract from specified or all layers
        layer_activations = extractor.extract_from_prompt_thought_pairs(
            prompt_thought_pairs,
            layer_indices=args.layers,
            show_progress=True
        )

        # Save activations if requested
        if not args.skip_save_activations:
            activations_path = output_dir / "activations.json"
            trainer = CAATrainer(args.model)
            trainer.save_activations(layer_activations, str(activations_path))

    # Step 3: Train CAA vectors
    print("\n[Step 3/5] Computing CAA steering vectors...")
    if not isinstance(trainer, CAATrainer):
        trainer = CAATrainer(args.model)

    vectors, metrics = trainer.train_all_layers(
        layer_activations,
        normalize=not args.no_normalize,
        verbose=True
    )

    # Step 4: Save results
    print("\n[Step 4/5] Saving results...")

    vectors_path = output_dir / "vectors.json"
    trainer.save_vectors(str(vectors_path))

    metrics_path = output_dir / "metrics.json"
    trainer.save_metrics(str(metrics_path))

    # Step 5: Generate visualizations
    if args.visualize:
        print(f"\n[Step 5/5] Generating visualizations...")
        visualizer = Visualizer(output_dir=str(plots_dir))

        # Main layer analysis
        print("  Creating layer analysis plot...")
        visualizer.plot_layer_analysis(
            metrics,
            model_name=args.model.split('/')[-1],
            save_name="layer_analysis.png"
        )

        # Find best layers, EXCLUDING first 4 and last 4 (usually bad for steering)
        all_layers = sorted(metrics.keys())
        num_layers = len(all_layers)

        # Define good layers (exclude first 4 and last 4)
        good_layers = [l for l in all_layers if l >= 4 and l < num_layers - 4]

        if not good_layers:
            print("  Warning: Not enough layers to exclude first/last 4. Using all layers.")
            good_layers = all_layers

        # Find best layer among good layers
        best_layer = max(
            [(l, metrics[l]['cohen_d']) for l in good_layers],
            key=lambda x: x[1]
        )[0]

        print(f"  Best layer for visualization: {best_layer} (excluding first/last 4)")

        # Distribution comparison for top layers (from good layers only)
        print(f"  Creating top {args.top_n} distribution plots (excluding first/last 4 layers)...")
        good_layer_metrics = {l: metrics[l] for l in good_layers}
        visualizer.plot_distribution_comparison(
            layer_activations,
            vectors,
            good_layer_metrics,  # Use filtered metrics
            top_n=args.top_n,
            save_name=f"top_{args.top_n}_distributions.png"
        )

        # Vector magnitudes
        print("  Creating vector magnitude plot...")
        visualizer.plot_vector_magnitudes(
            vectors,
            save_name="vector_magnitudes.png"
        )

        # Activation heatmap for best layer (with CAA vector)
        print(f"  Creating activation heatmap for layer {best_layer}...")
        print(f"    Showing top 500 most discriminative dimensions (out of {vectors[best_layer].shape[0]})...")
        visualizer.plot_activation_heatmap(
            layer_activations,
            layer=best_layer,
            caa_vector=vectors[best_layer],  # Add CAA vector for 3rd panel
            num_samples=min(30, len(layer_activations[best_layer]['high'])),
            num_dims=500,  # Show top 500 discriminative dimensions
            rank_by_separation=True,
            save_name=f"activation_heatmap_layer{best_layer}_top500dims.png"
        )

        # Optional: Create a version with ALL dimensions (if requested via env var or flag)
        import os
        if os.environ.get('SHOW_ALL_DIMS', '').lower() == 'true':
            print(f"  Creating full-dimension heatmap for layer {best_layer}...")
            print(f"    This will create a very large figure...")
            visualizer.plot_activation_heatmap(
                layer_activations,
                layer=best_layer,
                caa_vector=vectors[best_layer],
                num_samples=min(30, len(layer_activations[best_layer]['high'])),
                num_dims='all',  # ALL 3584 dimensions
                rank_by_separation=True,
                save_name=f"activation_heatmap_layer{best_layer}_all_dims.png"
            )

    # Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nOutput files:")
    print(f"  Vectors: {vectors_path}")
    print(f"  Metrics: {metrics_path}")
    if not args.skip_save_activations and not args.load_activations:
        print(f"  Activations: {output_dir / 'activations.json'}")
    if args.visualize:
        print(f"  Plots: {plots_dir}/")

    # Print best layers
    sorted_layers = sorted(metrics.items(), key=lambda x: x[1]['cohen_d'], reverse=True)
    print(f"\nBest layers (by Cohen's d):")
    for layer, m in sorted_layers[:5]:
        print(f"  Layer {layer:2d}: Cohen's d={m['cohen_d']:.3f}, "
              f"Accuracy={m['accuracy']:.3f}, p={m['p_value']:.2e}")

    print(f"\nRecommendation: Use layer {sorted_layers[0][0]} for steering")

    print("\nNext steps:")
    print(f"  1. Apply steering: python steering/apply_steering.py --vectors {vectors_path}")
    print(f"  2. Analyze layers: python steering/analyze_layers.py --metrics {metrics_path}")
    print(f"  3. Compare datasets: python steering/compare_vectors.py")


if __name__ == "__main__":
    main()
