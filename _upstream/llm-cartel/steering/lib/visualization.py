"""
Visualization Module

Creates comprehensive visualizations for analyzing CAA steering vectors
across layers, including separation quality, metrics heatmaps, and
distribution comparisons.
Enhanced with temporal analysis visualizations for turn-based dynamics.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import seaborn as sns
import warnings
from matplotlib import cm
from matplotlib.colors import Normalize

# Configure plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class Visualizer:
    """Create comprehensive visualizations for CAA analysis"""

    def __init__(self, output_dir: str = "plots"):
        """
        Initialize visualizer

        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_layer_analysis(self,
                           metrics: Dict[int, Dict],
                           model_name: str = "Model",
                           save_name: str = "layer_analysis.png"):
        """
        Create comprehensive 6-panel layer-wise analysis visualization

        Args:
            metrics: Dictionary mapping layer index to metrics dict
            model_name: Name of model for title
            save_name: Filename to save plot
        """
        layers = sorted(metrics.keys())

        # Extract metrics
        cohen_ds = [metrics[l]['cohen_d'] for l in layers]
        accuracies = [metrics[l]['accuracy'] for l in layers]
        fisher_ratios = [metrics[l]['fisher_ratio'] for l in layers]
        min_margins = [metrics[l]['min_margin'] for l in layers]
        mean_margins = [metrics[l]['mean_margin'] for l in layers]
        overlaps = [metrics[l]['overlap_percent'] for l in layers]
        aucs = [metrics[l]['auc_roc'] for l in layers]
        silhouettes = [metrics[l]['silhouette'] for l in layers]

        # Find top 3 layers
        n_top = min(3, len(layers))
        top_indices = np.argsort(cohen_ds)[-n_top:][::-1]
        top_layers = [layers[i] for i in top_indices]

        # Create figure
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle(f'Layer-wise CAA Analysis - {model_name}', fontsize=16, fontweight='bold')

        # Panel 1: Primary Metrics (Cohen's d and Accuracy)
        ax1 = plt.subplot(2, 3, 1)
        color1 = 'tab:blue'
        ax1.set_xlabel('Layer Index')
        ax1.set_ylabel("Cohen's d", color=color1)
        line1 = ax1.plot(layers, cohen_ds, color=color1, marker='o', label="Cohen's d", linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)

        # Mark top 3 layers
        for idx in top_indices:
            ax1.plot(layers[idx], cohen_ds[idx], marker='*', markersize=20,
                    color='gold', markeredgecolor='black', markeredgewidth=1.5)

        # Secondary y-axis for accuracy
        ax1_2 = ax1.twinx()
        color2 = 'tab:green'
        ax1_2.set_ylabel('Accuracy', color=color2)
        line2 = ax1_2.plot(layers, accuracies, color=color2, marker='s',
                          linestyle='--', alpha=0.7, label='Accuracy', linewidth=2)
        ax1_2.tick_params(axis='y', labelcolor=color2)

        # Legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        ax1.set_title('Primary Separation Metrics', fontweight='bold')

        # Panel 2: Robustness Metrics (Fisher ratio and min margin)
        ax2 = plt.subplot(2, 3, 2)

        # Fisher ratio as bars
        bars = ax2.bar(layers, fisher_ratios, alpha=0.7, color='skyblue', label='Fisher Ratio')

        # Color bars by quality
        colors = ['red' if f < 1 else 'yellow' if f < 5 else 'green' for f in fisher_ratios]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
            bar.set_alpha(0.7)

        ax2.set_xlabel('Layer Index')
        ax2.set_ylabel('Fisher Ratio', color='skyblue')
        ax2.set_title('Robustness Metrics', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='skyblue')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left')

        # Min margin as line on secondary axis
        ax2_2 = ax2.twinx()
        ax2_2.plot(layers, min_margins, color='darkred', marker='^',
                  linewidth=2, label='Min Margin')
        ax2_2.set_ylabel('Minimum Margin', color='darkred')
        ax2_2.tick_params(axis='y', labelcolor='darkred')
        ax2_2.legend(loc='upper right')

        # Panel 3: Cluster Quality Metrics
        ax3 = plt.subplot(2, 3, 3)
        ax3.plot(layers, [1 - o for o in overlaps], color='purple',
                marker='D', label='Separation (1 - Overlap)', linewidth=2)
        ax3.plot(layers, silhouettes, color='orange', marker='p',
                linestyle='--', label='Silhouette Score', alpha=0.8)
        ax3.plot(layers, aucs, color='teal', marker='h',
                linestyle=':', label='AUC-ROC', alpha=0.8)
        ax3.set_xlabel('Layer Index')
        ax3.set_ylabel('Score')
        ax3.set_title('Cluster Quality Metrics', fontweight='bold')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 1.05])

        # Panel 4: Comprehensive Metrics Heatmap
        ax4 = plt.subplot(2, 3, (4, 5))

        metric_names = ['Cohen\'s d', 'Accuracy', 'Fisher Ratio', 'AUC-ROC',
                       'Silhouette', 'Sep. (1-Overlap)', 'Min Margin', 'Mean Margin']

        # Prepare heatmap data
        heatmap_data = []
        for layer in layers:
            row = [
                metrics[layer]['cohen_d'],
                metrics[layer]['accuracy'],
                min(metrics[layer]['fisher_ratio'], 10),  # Cap for visualization
                metrics[layer]['auc_roc'],
                metrics[layer]['silhouette'],
                1 - metrics[layer]['overlap_percent'],
                min(metrics[layer]['min_margin'], 1),  # Cap for visualization
                min(metrics[layer]['mean_margin'], 1)   # Cap for visualization
            ]
            heatmap_data.append(row)

        # Normalize each metric to [0, 1]
        heatmap_array = np.array(heatmap_data)
        for i in range(heatmap_array.shape[1]):
            col = heatmap_array[:, i]
            if np.max(col) > np.min(col):
                heatmap_array[:, i] = (col - np.min(col)) / (np.max(col) - np.min(col))

        # Create heatmap
        im = ax4.imshow(heatmap_array.T, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
        ax4.set_xticks(np.arange(len(layers))[::max(1, len(layers)//20)])  # Show every Nth label
        ax4.set_xticklabels(layers[::max(1, len(layers)//20)], fontsize=8)
        ax4.set_yticks(np.arange(len(metric_names)))
        ax4.set_yticklabels(metric_names)

        # Highlight best layer
        best_layer_idx = np.argmax(cohen_ds)
        rect = plt.Rectangle((best_layer_idx - 0.5, -0.5), 1, len(metric_names),
                             fill=False, edgecolor='blue', linewidth=3)
        ax4.add_patch(rect)

        cbar = plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
        cbar.set_label('Normalized Score', rotation=270, labelpad=15)
        ax4.set_xlabel('Layer Index')
        ax4.set_title('Comprehensive Metrics Heatmap (Normalized)', fontweight='bold')

        # Panel 6: Summary & Recommendations
        ax6 = plt.subplot(2, 3, 6)

        summary_text = f'Top {n_top} Layers Summary\n\n'
        summary_text += f'Best Layers: {", ".join(map(str, top_layers))}\n\n'

        for i, layer in enumerate(top_layers):
            summary_text += f'Layer {layer}:\n'
            summary_text += f'  Cohen\'s d: {metrics[layer]["cohen_d"]:.3f}\n'
            summary_text += f'  Accuracy: {metrics[layer]["accuracy"]:.3f}\n'
            summary_text += f'  p-value: {metrics[layer]["p_value"]:.2e}\n\n'

        summary_text += f'\nRecommendation:\nUse Layer {top_layers[0]} for steering'

        ax6.text(0.5, 0.5, summary_text,
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax6.transAxes,
                fontsize=11,
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax6.set_xlim([0, 1])
        ax6.set_ylim([0, 1])
        ax6.axis('off')
        ax6.set_title('Summary & Recommendations', fontweight='bold')

        # Save figure
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved layer analysis plot to {save_path}")
        plt.close()

    def plot_distribution_comparison(self,
                                    activations: Dict[int, Dict[str, List]],
                                    vectors: Dict[int, np.ndarray],
                                    metrics: Dict[int, Dict],
                                    top_n: int = 5,
                                    save_name: str = "top_distributions.png"):
        """
        Create distribution plots for top N layers

        Args:
            activations: Dictionary of activations {layer: {'high': [...], 'low': [...]}}
            vectors: Dictionary of CAA vectors
            metrics: Dictionary of metrics
            top_n: Number of top layers to visualize
            save_name: Filename to save plot
        """
        # Find top N layers by Cohen's d
        sorted_layers = sorted(metrics.items(), key=lambda x: x[1]['cohen_d'], reverse=True)
        top_layers = [l[0] for l in sorted_layers[:top_n]]

        # Create figure
        fig, axes = plt.subplots(1, top_n, figsize=(4*top_n, 4))
        if top_n == 1:
            axes = [axes]

        for idx, layer in enumerate(top_layers):
            ax = axes[idx]

            # Get activations
            high_acts = np.stack([
                act.cpu().numpy() if isinstance(act, torch.Tensor) else act
                for act in activations[layer]['high']
            ])
            low_acts = np.stack([
                act.cpu().numpy() if isinstance(act, torch.Tensor) else act
                for act in activations[layer]['low']
            ])

            # Get vector
            caa_vector = vectors[layer]

            # Project onto CAA vector
            high_proj = high_acts @ caa_vector
            low_proj = low_acts @ caa_vector

            # Plot distributions
            ax.hist(low_proj, bins=30, alpha=0.6, color='blue',
                   label='Low Collusion', density=True, edgecolor='darkblue')
            ax.hist(high_proj, bins=30, alpha=0.6, color='red',
                   label='High Collusion', density=True, edgecolor='darkred')

            # Add decision boundary
            threshold = (np.mean(high_proj) + np.mean(low_proj)) / 2
            ax.axvline(x=threshold, color='black', linestyle='--',
                      linewidth=2, label='Decision Boundary')

            # Add vertical lines for means
            ax.axvline(x=np.mean(high_proj), color='darkred', linestyle=':',
                      alpha=0.7, linewidth=2)
            ax.axvline(x=np.mean(low_proj), color='darkblue', linestyle=':',
                      alpha=0.7, linewidth=2)

            # Add metrics text
            m = metrics[layer]
            metrics_text = (f"Cohen's d: {m['cohen_d']:.3f}\n"
                          f"Accuracy: {m['accuracy']:.3f}\n"
                          f"AUC-ROC: {m['auc_roc']:.3f}\n"
                          f"p-value: {m['p_value']:.2e}")

            ax.text(0.02, 0.98, metrics_text,
                   transform=ax.transAxes,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                   fontsize=10,
                   fontfamily='monospace')

            ax.set_xlabel('CAA Projection Value', fontsize=11)
            ax.set_ylabel('Density', fontsize=11)
            ax.set_title(f'Layer {layer}', fontweight='bold', fontsize=13)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f'Projection Distributions - Top {top_n} Layers',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved distribution comparison to {save_path}")
        plt.close()

    def plot_activation_heatmap(self,
                               activations: Dict[int, Dict[str, List]],
                               layer: int,
                               caa_vector: Optional[np.ndarray] = None,
                               num_samples: int = 20,
                               num_dims: int = 500,
                               rank_by_separation: bool = True,
                               save_name: str = "activation_heatmap.png"):
        """
        Create comprehensive heatmap of activations for a specific layer

        Args:
            activations: Dictionary of activations
            layer: Layer to visualize
            caa_vector: CAA vector for this layer (optional, for 3rd panel)
            num_samples: Number of samples to show
            num_dims: Number of dimensions to show (or 'all' for all dimensions)
            rank_by_separation: If True, show top N most discriminative dimensions
            save_name: Filename to save plot
        """
        # Get activations
        high_acts = np.stack([
            act.cpu().numpy() if isinstance(act, torch.Tensor) else act
            for act in activations[layer]['high'][:num_samples]
        ])
        low_acts = np.stack([
            act.cpu().numpy() if isinstance(act, torch.Tensor) else act
            for act in activations[layer]['low'][:num_samples]
        ])

        total_dims = high_acts.shape[1]

        # Dimension selection strategy
        if rank_by_separation:
            # Rank dimensions by discriminative power
            high_mean_all = high_acts.mean(axis=0)
            low_mean_all = low_acts.mean(axis=0)
            separation = np.abs(high_mean_all - low_mean_all)

            # Get top N dimensions
            if num_dims == 'all' or num_dims >= total_dims:
                top_dims = np.argsort(separation)[::-1]  # All dims, sorted by importance
                num_dims = total_dims
            else:
                top_dims = np.argsort(separation)[-num_dims:][::-1]  # Top N, most important first

            high_acts_selected = high_acts[:, top_dims]
            low_acts_selected = low_acts[:, top_dims]
            dim_labels = top_dims
            title_suffix = f'Top {num_dims} Most Discriminative Dimensions'
        else:
            # Show first N dimensions
            num_dims = min(num_dims if num_dims != 'all' else total_dims, total_dims)
            high_acts_selected = high_acts[:, :num_dims]
            low_acts_selected = low_acts[:, :num_dims]
            dim_labels = np.arange(num_dims)
            title_suffix = f'First {num_dims} Dimensions'

        # Combine samples
        all_acts = np.vstack([high_acts_selected, low_acts_selected])

        # Calculate figure height based on number of dimensions
        # Rule: ~0.05 inches per dimension, min 8 inches, max 100 inches
        fig_height = max(8, min(100, num_dims * 0.05))

        # Create figure with 3 panels if CAA vector provided, else 2 panels
        n_panels = 3 if caa_vector is not None else 2
        fig_width = 8 * n_panels
        fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, fig_height))

        if n_panels == 2:
            axes = list(axes)
            axes.append(None)  # Dummy for indexing

        # Panel 1: Heatmap
        ax = axes[0]
        im = ax.imshow(all_acts.T, aspect='auto', cmap='RdBu_r', interpolation='nearest')
        ax.set_xlabel('Sample Index', fontsize=11)
        ax.set_ylabel('Activation Dimension (ranked by separation)', fontsize=11)
        ax.set_title(f'Layer {layer} - Activation Patterns\n{title_suffix}',
                    fontweight='bold', fontsize=12)

        # Add separator and labels
        ax.axvline(x=num_samples-0.5, color='black', linewidth=2)
        ax.text(num_samples/2, -num_dims*0.02, 'High Collusion',
               ha='center', fontweight='bold', fontsize=10)
        ax.text(num_samples*1.5, -num_dims*0.02, 'Low Collusion',
               ha='center', fontweight='bold', fontsize=10)

        # Colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Add dimension indices every N dimensions for reference
        tick_spacing = max(1, num_dims // 20)  # ~20 ticks
        tick_positions = np.arange(0, num_dims, tick_spacing)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels([f'{dim_labels[i]}' for i in tick_positions], fontsize=8)

        # Panel 2: Mean patterns with shading
        ax = axes[1]
        high_mean = high_acts_selected.mean(axis=0)
        low_mean = low_acts_selected.mean(axis=0)
        difference = high_mean - low_mean

        x = np.arange(len(high_mean))

        # Plot means
        ax.plot(x, high_mean, 'r-', label='High Collusion Mean',
               alpha=0.8, linewidth=1.5)
        ax.plot(x, low_mean, 'b-', label='Low Collusion Mean',
               alpha=0.8, linewidth=1.5)

        # Shade regions where high > low (red) and low > high (blue)
        ax.fill_between(x, high_mean, low_mean,
                       where=(high_mean >= low_mean),
                       interpolate=True, alpha=0.3, color='red',
                       label='High > Low')
        ax.fill_between(x, high_mean, low_mean,
                       where=(high_mean < low_mean),
                       interpolate=True, alpha=0.3, color='blue',
                       label='Low > High')

        ax.set_xlabel('Dimension Index (ranked)', fontsize=11)
        ax.set_ylabel('Mean Activation Value', fontsize=11)
        ax.set_title('Mean Activation Patterns', fontweight='bold', fontsize=12)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)

        # Add statistics text
        mean_separation = np.mean(np.abs(difference))
        max_separation = np.max(np.abs(difference))
        stats_text = (f'Mean |separation|: {mean_separation:.3f}\n'
                     f'Max |separation|: {max_separation:.3f}')
        ax.text(0.02, 0.98, stats_text,
               transform=ax.transAxes,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
               fontsize=9, fontfamily='monospace')

        # Panel 3: CAA vector components (if provided)
        if caa_vector is not None and axes[2] is not None:
            ax = axes[2]

            # Get CAA components for selected dimensions
            if rank_by_separation:
                caa_selected = caa_vector[top_dims]
            else:
                caa_selected = caa_vector[:num_dims]

            # Plot CAA vector components
            x = np.arange(len(caa_selected))

            # Color by sign
            colors = ['red' if c > 0 else 'blue' for c in caa_selected]
            ax.barh(x, caa_selected, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

            ax.set_ylabel('Dimension Index (ranked)', fontsize=11)
            ax.set_xlabel('CAA Vector Component Value', fontsize=11)
            ax.set_title('CAA Vector Components\n(Red: favors high, Blue: favors low)',
                        fontweight='bold', fontsize=12)
            ax.grid(True, alpha=0.3, axis='x')
            ax.axvline(x=0, color='black', linewidth=1)

            # Add statistics
            caa_stats = (f'Mean |component|: {np.mean(np.abs(caa_selected)):.4f}\n'
                        f'Max |component|: {np.max(np.abs(caa_selected)):.4f}\n'
                        f'L2 norm: {np.linalg.norm(caa_selected):.4f}')
            ax.text(0.98, 0.02, caa_stats,
                   transform=ax.transAxes,
                   verticalalignment='bottom',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                   fontsize=9, fontfamily='monospace')

        plt.tight_layout()

        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=200, bbox_inches='tight')  # Higher DPI for tall figures
        print(f"Saved activation heatmap to {save_path}")
        print(f"  Figure size: {fig_width} x {fig_height} inches")
        print(f"  Dimensions shown: {num_dims}/{total_dims}")
        plt.close()

    def plot_vector_magnitudes(self,
                              vectors: Dict[int, np.ndarray],
                              save_name: str = "vector_magnitudes.png"):
        """
        Plot vector magnitudes across layers

        Args:
            vectors: Dictionary of CAA vectors
            save_name: Filename to save plot
        """
        layers = sorted(vectors.keys())
        magnitudes = [np.linalg.norm(vectors[l]) for l in layers]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(layers, magnitudes, marker='o', linewidth=2, markersize=8, color='steelblue')
        ax.set_xlabel('Layer Index', fontsize=12)
        ax.set_ylabel('Vector Magnitude', fontsize=12)
        ax.set_title('CAA Steering Vector Magnitudes Across Layers', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Check if vectors are normalized (all close to 1.0)
        all_normalized = all(abs(m - 1.0) < 0.01 for m in magnitudes)

        if all_normalized:
            # Vectors are normalized - set sensible y-axis limits
            # Show range [0.95, 1.05] to make it clear they're all ~1.0
            ax.set_ylim([0.95, 1.05])
            ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Unit Norm (1.0)')
            ax.legend()

            # Add text annotation
            ax.text(0.02, 0.98,
                   f'All vectors normalized to unit length\n'
                   f'Mean magnitude: {np.mean(magnitudes):.10f}\n'
                   f'Std deviation: {np.std(magnitudes):.2e}',
                   transform=ax.transAxes,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   fontsize=10,
                   fontfamily='monospace')
        else:
            # Vectors not normalized - show full range
            ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Unit Norm')
            ax.legend()

        plt.tight_layout()

        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved vector magnitudes plot to {save_path}")
        plt.close()

    def plot_temporal_evolution(self,
                               temporal_vectors: Dict[str, np.ndarray],
                               separation_metrics: Dict[str, Dict],
                               layer: int,
                               save_name: str = "temporal_evolution.png"):
        """
        Plot temporal evolution of steering vectors across game turns

        Args:
            temporal_vectors: Dictionary of vectors by temporal segment
            separation_metrics: Separation quality metrics by segment
            layer: Layer index for title
            save_name: Filename to save plot
        """
        segment_names = list(temporal_vectors.keys())
        n_segments = len(segment_names)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Temporal Evolution of Steering Vectors - Layer {layer}', fontsize=16, fontweight='bold')

        # Panel 1: Cohen's d evolution
        ax1 = axes[0, 0]
        cohen_ds = [separation_metrics[name]['cohen_d'] for name in segment_names]
        accuracies = [separation_metrics[name]['accuracy'] for name in segment_names]

        x = np.arange(n_segments)
        ax1.bar(x - 0.2, cohen_ds, width=0.4, label="Cohen's d", color='steelblue')
        ax1.set_ylabel("Cohen's d", color='steelblue')
        ax1.tick_params(axis='y', labelcolor='steelblue')

        ax1_2 = ax1.twinx()
        ax1_2.bar(x + 0.2, accuracies, width=0.4, label='Accuracy', color='coral', alpha=0.7)
        ax1_2.set_ylabel('Accuracy', color='coral')
        ax1_2.tick_params(axis='y', labelcolor='coral')

        ax1.set_xticks(x)
        ax1.set_xticklabels(segment_names)
        ax1.set_title('Separation Quality Over Time')
        ax1.grid(True, alpha=0.3)

        # Panel 2: Vector similarity matrix
        ax2 = axes[0, 1]
        n = len(segment_names)
        similarity_matrix = np.zeros((n, n))

        for i, name1 in enumerate(segment_names):
            for j, name2 in enumerate(segment_names):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    vec1 = temporal_vectors[name1]
                    vec2 = temporal_vectors[name2]
                    similarity_matrix[i, j] = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

        im = ax2.imshow(similarity_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax2.set_xticks(range(n))
        ax2.set_yticks(range(n))
        ax2.set_xticklabels(segment_names, rotation=45)
        ax2.set_yticklabels(segment_names)
        ax2.set_title('Vector Similarity Matrix')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Cosine Similarity')

        # Add text annotations
        for i in range(n):
            for j in range(n):
                text = ax2.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                              ha="center", va="center", color="black" if abs(similarity_matrix[i, j]) < 0.5 else "white")

        # Panel 3: Angular drift
        ax3 = axes[1, 0]
        if n_segments > 1:
            angles = []
            for i in range(n_segments - 1):
                vec1 = temporal_vectors[segment_names[i]]
                vec2 = temporal_vectors[segment_names[i + 1]]
                cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                angle = np.arccos(np.clip(cos_sim, -1, 1)) * 180 / np.pi
                angles.append(angle)

            transitions = [f"{segment_names[i]}→{segment_names[i+1]}" for i in range(n_segments - 1)]
            ax3.plot(transitions, angles, marker='o', markersize=10, linewidth=2, color='darkgreen')
            ax3.set_ylabel('Angular Distance (degrees)')
            ax3.set_title('Sequential Vector Drift')
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(bottom=0)

            # Add horizontal lines for reference
            ax3.axhline(y=30, color='orange', linestyle='--', alpha=0.5, label='30° (moderate drift)')
            ax3.axhline(y=60, color='red', linestyle='--', alpha=0.5, label='60° (high drift)')
            ax3.legend()

        # Panel 4: Sample size distribution
        ax4 = axes[1, 1]
        n_samples = [separation_metrics[name]['n_samples'] for name in segment_names]
        bars = ax4.bar(segment_names, n_samples, color='mediumpurple')
        ax4.set_ylabel('Number of Samples')
        ax4.set_title('Sample Distribution Across Time')
        ax4.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, n in zip(bars, n_samples):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(n)}', ha='center', va='bottom')

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved temporal evolution plot to {save_path}")
        plt.close()

    def plot_activation_trajectories(self,
                                    trajectories: Dict[str, List[Dict]],
                                    layer: int,
                                    episode_ids: Optional[List] = None,
                                    save_name: str = "activation_trajectories.png"):
        """
        Plot activation trajectories over turns within episodes

        Args:
            trajectories: Dictionary with 'high' and 'low' trajectory data
            layer: Layer index for title
            episode_ids: Optional list of episode IDs for labeling
            save_name: Filename to save plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Activation Trajectories - Layer {layer}', fontsize=16, fontweight='bold')

        # Plot high collusion trajectories
        ax1 = axes[0]
        if trajectories['high']:
            steps = [t['step'] for t in trajectories['high']]
            projections = [t['projection'] for t in trajectories['high']]
            ax1.plot(steps, projections, marker='o', label='High Collusion', color='red', alpha=0.7)

        if trajectories['low']:
            steps = [t['step'] for t in trajectories['low']]
            projections = [t['projection'] for t in trajectories['low']]
            ax1.plot(steps, projections, marker='s', label='Low Collusion', color='blue', alpha=0.7)

        ax1.set_xlabel('Turn Number')
        ax1.set_ylabel('CAA Projection')
        ax1.set_title('Projection Evolution Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot projection distributions
        ax2 = axes[1]
        if trajectories['high']:
            high_proj = [t['projection'] for t in trajectories['high']]
            ax2.hist(high_proj, bins=30, alpha=0.5, color='red', label='High Collusion', density=True)

        if trajectories['low']:
            low_proj = [t['projection'] for t in trajectories['low']]
            ax2.hist(low_proj, bins=30, alpha=0.5, color='blue', label='Low Collusion', density=True)

        ax2.set_xlabel('CAA Projection Value')
        ax2.set_ylabel('Density')
        ax2.set_title('Projection Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved activation trajectories plot to {save_path}")
        plt.close()

    def plot_turn_gradient_embeddings(self,
                                     embeddings_data: Dict[str, Any],
                                     layer: int,
                                     save_name: str = "turn_gradient_embeddings.png"):
        """
        Plot 2D embeddings colored by turn number gradient

        Args:
            embeddings_data: Output from TemporalAnalyzer.compute_embedding_evolution
            layer: Layer index for title
            save_name: Filename to save plot
        """
        embeddings = embeddings_data['embeddings']
        labels = embeddings_data['labels']
        steps = embeddings_data['steps']

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(f'Temporal Evolution in Embedding Space - Layer {layer}', fontsize=16, fontweight='bold')

        # Normalize steps for color mapping
        norm = Normalize(vmin=steps.min(), vmax=steps.max())
        cmap = cm.get_cmap('viridis')

        # Panel 1: All points colored by turn number
        ax1 = axes[0]
        scatter = ax1.scatter(embeddings[:, 0], embeddings[:, 1],
                            c=steps, cmap='viridis', alpha=0.6, s=30)
        ax1.set_xlabel('Component 1')
        ax1.set_ylabel('Component 2')
        ax1.set_title('Embeddings Colored by Turn Number')

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Turn Number')

        # Panel 2: Separated by class, colored by turn
        ax2 = axes[1]

        # Plot high collusion
        high_mask = labels == 1
        ax2.scatter(embeddings[high_mask, 0], embeddings[high_mask, 1],
                   c=steps[high_mask], cmap='Reds', alpha=0.6, s=30,
                   label='High Collusion', vmin=steps.min(), vmax=steps.max())

        # Plot low collusion
        low_mask = labels == 0
        ax2.scatter(embeddings[low_mask, 0], embeddings[low_mask, 1],
                   c=steps[low_mask], cmap='Blues', alpha=0.6, s=30,
                   label='Low Collusion', vmin=steps.min(), vmax=steps.max())

        ax2.set_xlabel('Component 1')
        ax2.set_ylabel('Component 2')
        ax2.set_title('Class-Separated Temporal Embeddings')
        ax2.legend()

        # Add arrow showing temporal direction (optional)
        # Group by turn ranges and draw arrows
        n_arrows = 5
        turn_bins = np.linspace(steps.min(), steps.max(), n_arrows + 1)

        for i in range(n_arrows - 1):
            mask = (steps >= turn_bins[i]) & (steps < turn_bins[i + 1])
            if mask.sum() > 0:
                center = embeddings[mask].mean(axis=0)

                next_mask = (steps >= turn_bins[i + 1]) & (steps < turn_bins[i + 2])
                if next_mask.sum() > 0:
                    next_center = embeddings[next_mask].mean(axis=0)
                    ax1.annotate('', xy=next_center, xytext=center,
                               arrowprops=dict(arrowstyle='->', lw=2, alpha=0.3, color='gray'))

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved turn gradient embeddings plot to {save_path}")
        plt.close()

    def plot_temporal_heatmap(self,
                             separation_metrics_by_layer: Dict[int, Dict[str, Dict]],
                             save_name: str = "temporal_heatmap.png"):
        """
        Create heatmap showing separation quality across layers and time segments

        Args:
            separation_metrics_by_layer: {layer: {segment: metrics}}
            save_name: Filename to save plot
        """
        # Extract layers and segments
        layers = sorted(separation_metrics_by_layer.keys())
        first_layer_metrics = separation_metrics_by_layer[layers[0]]
        segments = list(first_layer_metrics.keys())

        # Create matrices for different metrics
        cohen_d_matrix = np.zeros((len(layers), len(segments)))
        accuracy_matrix = np.zeros((len(layers), len(segments)))

        for i, layer in enumerate(layers):
            for j, segment in enumerate(segments):
                metrics = separation_metrics_by_layer[layer][segment]
                cohen_d_matrix[i, j] = metrics['cohen_d']
                accuracy_matrix[i, j] = metrics['accuracy']

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Temporal Separation Quality Across Layers', fontsize=16, fontweight='bold')

        # Panel 1: Cohen's d heatmap
        ax1 = axes[0]
        im1 = ax1.imshow(cohen_d_matrix, aspect='auto', cmap='YlOrRd')
        ax1.set_xticks(range(len(segments)))
        ax1.set_yticks(range(len(layers)))
        ax1.set_xticklabels(segments)
        ax1.set_yticklabels([f"Layer {l}" for l in layers])
        ax1.set_xlabel('Temporal Segment')
        ax1.set_ylabel('Layer')
        ax1.set_title("Cohen's d (Effect Size)")

        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label("Cohen's d")

        # Add text annotations
        for i in range(len(layers)):
            for j in range(len(segments)):
                text = ax1.text(j, i, f'{cohen_d_matrix[i, j]:.2f}',
                              ha="center", va="center",
                              color="white" if cohen_d_matrix[i, j] > cohen_d_matrix.max() * 0.5 else "black")

        # Panel 2: Accuracy heatmap
        ax2 = axes[1]
        im2 = ax2.imshow(accuracy_matrix, aspect='auto', cmap='RdYlGn', vmin=0.5, vmax=1.0)
        ax2.set_xticks(range(len(segments)))
        ax2.set_yticks(range(len(layers)))
        ax2.set_xticklabels(segments)
        ax2.set_yticklabels([f"Layer {l}" for l in layers])
        ax2.set_xlabel('Temporal Segment')
        ax2.set_ylabel('Layer')
        ax2.set_title('Classification Accuracy')

        # Add colorbar
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label('Accuracy')

        # Add text annotations
        for i in range(len(layers)):
            for j in range(len(segments)):
                text = ax2.text(j, i, f'{accuracy_matrix[i, j]:.1%}',
                              ha="center", va="center",
                              color="white" if accuracy_matrix[i, j] > 0.75 else "black")

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved temporal heatmap to {save_path}")
        plt.close()


if __name__ == "__main__":
    print("Visualization Module")
    print("Use this module as part of the training pipeline")
