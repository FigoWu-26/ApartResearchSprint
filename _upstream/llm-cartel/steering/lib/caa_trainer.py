"""
CAA Trainer Module

Computes Contrastive Activation Addition (CAA) steering vectors and evaluates
their effectiveness using comprehensive metrics.
Enhanced with temporal metadata support for turn-based analysis.
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import roc_auc_score, silhouette_score
from scipy import stats
import warnings


class CAATrainer:
    """Compute CAA steering vectors and evaluate their performance"""

    def __init__(self, model_name: str, extraction_method: str = "thought_tokens_only"):
        """
        Initialize the CAA trainer

        Args:
            model_name: Name of the model used for extraction
            extraction_method: Method used for activation extraction
        """
        self.model_name = model_name
        self.extraction_method = extraction_method
        self.vectors: Dict[int, np.ndarray] = {}
        self.metrics: Dict[int, Dict] = {}

    def compute_caa_vector(self,
                          high_activations: np.ndarray,
                          low_activations: np.ndarray,
                          normalize: bool = True) -> np.ndarray:
        """
        Compute CAA steering vector using mean difference method

        Args:
            high_activations: High collusion activations (n_samples, hidden_dim)
            low_activations: Low collusion activations (n_samples, hidden_dim)
            normalize: Whether to normalize vector to unit length

        Returns:
            CAA steering vector
        """
        # Compute mean of each class
        mean_high = np.mean(high_activations, axis=0)
        mean_low = np.mean(low_activations, axis=0)

        # CAA vector is the difference
        caa_vector = mean_high - mean_low

        # Normalize if requested
        if normalize:
            norm = np.linalg.norm(caa_vector)
            if norm > 0:
                caa_vector = caa_vector / norm
            else:
                warnings.warn("CAA vector has zero norm - returning unnormalized")

        return caa_vector

    def compute_layer_metrics(self,
                             high_activations: np.ndarray,
                             low_activations: np.ndarray,
                             caa_vector: np.ndarray) -> Dict:
        """
        Compute comprehensive separation metrics for a single layer

        Args:
            high_activations: High collusion activations
            low_activations: Low collusion activations
            caa_vector: CAA steering vector

        Returns:
            Dictionary with all computed metrics
        """
        # Project activations onto CAA vector
        high_proj = high_activations @ caa_vector
        low_proj = low_activations @ caa_vector

        # Basic statistics
        mean_high = np.mean(high_proj)
        mean_low = np.mean(low_proj)
        std_high = np.std(high_proj)
        std_low = np.std(low_proj)
        var_high = np.var(high_proj)
        var_low = np.var(low_proj)

        # Cohen's d (effect size) - standardized mean difference
        pooled_std = np.sqrt((std_high**2 + std_low**2) / 2)
        cohen_d = abs(mean_high - mean_low) / (pooled_std + 1e-8)

        # Fisher's Discriminant Ratio (proper signal-to-noise ratio)
        fisher_ratio = (mean_high - mean_low)**2 / (var_high + var_low + 1e-8)

        # Classification accuracy with optimal threshold
        threshold = (mean_high + mean_low) / 2
        correct_high = np.sum(high_proj > threshold) if mean_high > mean_low else np.sum(high_proj < threshold)
        correct_low = np.sum(low_proj < threshold) if mean_high > mean_low else np.sum(low_proj > threshold)
        accuracy = (correct_high + correct_low) / (len(high_proj) + len(low_proj))

        # AUC-ROC
        try:
            labels = np.concatenate([np.ones(len(high_proj)), np.zeros(len(low_proj))])
            scores = np.concatenate([high_proj, low_proj])
            auc_roc = roc_auc_score(labels, scores)
        except Exception as e:
            warnings.warn(f"Could not compute AUC-ROC: {e}")
            auc_roc = 0.5

        # Silhouette coefficient (clustering quality)
        try:
            all_proj = np.concatenate([high_proj, low_proj])
            all_labels = np.concatenate([np.ones(len(high_proj)), np.zeros(len(low_proj))])
            silhouette = silhouette_score(all_proj.reshape(-1, 1), all_labels)
        except Exception as e:
            warnings.warn(f"Could not compute silhouette score: {e}")
            silhouette = 0.0

        # Overlap percentage (misclassified samples)
        overlap_high = np.sum(high_proj < threshold) if mean_high > mean_low else np.sum(high_proj > threshold)
        overlap_low = np.sum(low_proj > threshold) if mean_high > mean_low else np.sum(low_proj < threshold)
        overlap_percent = (overlap_high + overlap_low) / (len(high_proj) + len(low_proj))

        # Margin metrics (robustness to perturbations)
        if mean_high > mean_low:
            min_margin_high = np.min(high_proj - threshold) if len(high_proj) > 0 else 0
            min_margin_low = np.min(threshold - low_proj) if len(low_proj) > 0 else 0
            mean_margin = (np.mean(high_proj - threshold) + np.mean(threshold - low_proj)) / 2
        else:
            min_margin_high = np.min(threshold - high_proj) if len(high_proj) > 0 else 0
            min_margin_low = np.min(low_proj - threshold) if len(low_proj) > 0 else 0
            mean_margin = (np.mean(threshold - high_proj) + np.mean(low_proj - threshold)) / 2

        min_margin = min(abs(min_margin_high), abs(min_margin_low))

        # T-statistic for statistical significance
        try:
            t_stat, p_value = stats.ttest_ind(high_proj, low_proj)
            t_stat = abs(t_stat)
        except Exception as e:
            warnings.warn(f"Could not compute t-test: {e}")
            t_stat = 0.0
            p_value = 1.0

        return {
            'cohen_d': float(cohen_d),
            'fisher_ratio': float(fisher_ratio),
            'accuracy': float(accuracy),
            'auc_roc': float(auc_roc),
            'silhouette': float(silhouette),
            'overlap_percent': float(overlap_percent),
            'min_margin': float(min_margin),
            'mean_margin': float(mean_margin),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'mean_high': float(mean_high),
            'mean_low': float(mean_low),
            'std_high': float(std_high),
            'std_low': float(std_low),
            'separation': float(abs(mean_high - mean_low)),
            'n_high': int(len(high_proj)),
            'n_low': int(len(low_proj))
        }

    def train_all_layers(self,
                        layer_activations: Dict[int, Dict[str, List[torch.Tensor]]],
                        normalize: bool = True,
                        verbose: bool = True) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict]]:
        """
        Compute CAA vectors and metrics for all layers

        Args:
            layer_activations: Dictionary with structure {layer: {'high': [...], 'low': [...]}}
            normalize: Whether to normalize vectors to unit length
            verbose: Whether to print progress

        Returns:
            Tuple of (vectors dict, metrics dict)
        """
        layers = sorted(layer_activations.keys())

        if verbose:
            print(f"\nComputing CAA vectors for {len(layers)} layers...")
            print(f"Normalization: {normalize}")

        vectors = {}
        metrics = {}

        for layer in layers:
            # Convert tensors to numpy arrays in float32 for numerical stability
            # CRITICAL: float16/bfloat16 can cause precision loss in dot products!
            high_acts = np.stack([
                act.cpu().numpy().astype(np.float32) if isinstance(act, torch.Tensor)
                else np.asarray(act, dtype=np.float32)
                for act in layer_activations[layer]['high']
            ])
            low_acts = np.stack([
                act.cpu().numpy().astype(np.float32) if isinstance(act, torch.Tensor)
                else np.asarray(act, dtype=np.float32)
                for act in layer_activations[layer]['low']
            ])

            # Compute CAA vector
            caa_vector = self.compute_caa_vector(high_acts, low_acts, normalize=normalize)
            vectors[layer] = caa_vector

            # Compute metrics
            layer_metrics = self.compute_layer_metrics(high_acts, low_acts, caa_vector)
            metrics[layer] = layer_metrics

            if verbose:
                print(f"  Layer {layer:2d}: Cohen's d={layer_metrics['cohen_d']:.3f}, "
                      f"Accuracy={layer_metrics['accuracy']:.3f}, "
                      f"Fisher={layer_metrics['fisher_ratio']:.3f}")

        # Store results
        self.vectors = vectors
        self.metrics = metrics

        # Print summary
        if verbose:
            self._print_summary()

        return vectors, metrics

    def _print_summary(self):
        """Print summary of training results"""
        if not self.metrics:
            print("No metrics to summarize")
            return

        # Find best layers by different criteria
        sorted_by_cohen = sorted(self.metrics.items(), key=lambda x: x[1]['cohen_d'], reverse=True)
        sorted_by_accuracy = sorted(self.metrics.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        sorted_by_fisher = sorted(self.metrics.items(), key=lambda x: x[1]['fisher_ratio'], reverse=True)

        print("\n" + "="*70)
        print("TRAINING SUMMARY")
        print("="*70)

        print(f"\nTop 5 layers by Cohen's d:")
        for layer, m in sorted_by_cohen[:5]:
            print(f"  Layer {layer:2d}: {m['cohen_d']:.3f} "
                  f"(accuracy={m['accuracy']:.3f}, p={m['p_value']:.2e})")

        print(f"\nTop 5 layers by accuracy:")
        for layer, m in sorted_by_accuracy[:5]:
            print(f"  Layer {layer:2d}: {m['accuracy']:.3f} "
                  f"(Cohen's d={m['cohen_d']:.3f})")

        print(f"\nTop 5 layers by Fisher ratio:")
        for layer, m in sorted_by_fisher[:5]:
            print(f"  Layer {layer:2d}: {m['fisher_ratio']:.3f} "
                  f"(Cohen's d={m['cohen_d']:.3f})")

        # Overall statistics
        cohen_ds = [m['cohen_d'] for m in self.metrics.values()]
        print(f"\nOverall statistics:")
        print(f"  Mean Cohen's d: {np.mean(cohen_ds):.3f} ± {np.std(cohen_ds):.3f}")
        print(f"  Max Cohen's d: {np.max(cohen_ds):.3f} (layer {sorted_by_cohen[0][0]})")
        print(f"  Min Cohen's d: {np.min(cohen_ds):.3f}")

    def save_vectors(self, output_path: str):
        """
        Save steering vectors to JSON file

        Args:
            output_path: Path to save vectors
        """
        if not self.vectors:
            warnings.warn("No vectors to save")
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            'model_name': self.model_name,
            'extraction_method': self.extraction_method,
            'context': 'full_prompt_plus_thought',
            'normalized': True,
            'num_layers': len(self.vectors),
            'vectors': {
                str(layer): vector.tolist()
                for layer, vector in self.vectors.items()
            }
        }

        with open(output_path, 'w') as f:
            json.dump(save_dict, f, indent=2)

        print(f"\nSaved steering vectors to {output_path}")

    def save_metrics(self, output_path: str):
        """
        Save metrics to JSON file

        Args:
            output_path: Path to save metrics
        """
        if not self.metrics:
            warnings.warn("No metrics to save")
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Find best layers
        sorted_layers = sorted(
            self.metrics.items(),
            key=lambda x: x[1]['cohen_d'],
            reverse=True
        )

        save_dict = {
            'model_name': self.model_name,
            'extraction_method': self.extraction_method,
            'num_layers': len(self.metrics),
            'best_layer': int(sorted_layers[0][0]),
            'top_5_layers': [int(l[0]) for l in sorted_layers[:5]],
            'layer_metrics': {
                str(layer): metrics
                for layer, metrics in self.metrics.items()
            }
        }

        with open(output_path, 'w') as f:
            json.dump(save_dict, f, indent=2)

        print(f"Saved metrics to {output_path}")

    def save_activations(self,
                        layer_activations: Dict[int, Dict[str, List[torch.Tensor]]],
                        output_path: str,
                        metadata_lists: Optional[Dict[str, List[Dict]]] = None):
        """
        Save extracted activations to JSON file with optional metadata

        Args:
            layer_activations: Dictionary of activations
            output_path: Path to save activations
            metadata_lists: Optional dictionary of metadata lists {'high': [...], 'low': [...]}
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert tensors to lists
        activations_to_save = {}
        for layer, acts in layer_activations.items():
            activations_to_save[str(layer)] = {
                'high': [
                    t.cpu().numpy().tolist() if isinstance(t, torch.Tensor) else t.tolist()
                    for t in acts['high']
                ],
                'low': [
                    t.cpu().numpy().tolist() if isinstance(t, torch.Tensor) else t.tolist()
                    for t in acts['low']
                ]
            }

        save_dict = {
            'model_name': self.model_name,
            'extraction_method': self.extraction_method,
            'context': 'full_prompt_plus_thought',
            'num_layers': len(layer_activations),
            'num_pairs': len(layer_activations[list(layer_activations.keys())[0]]['high']),
            'activations': activations_to_save
        }

        # Add temporal metadata if provided
        if metadata_lists is not None:
            save_dict['temporal_metadata'] = metadata_lists

            # Add summary statistics
            if metadata_lists.get('high') and 'step' in metadata_lists['high'][0]:
                steps_high = [m['step'] for m in metadata_lists['high'] if m.get('step') is not None]
                steps_low = [m['step'] for m in metadata_lists['low'] if m.get('step') is not None]
                if steps_high and steps_low:
                    save_dict['temporal_summary'] = {
                        'high_step_range': [min(steps_high), max(steps_high)],
                        'low_step_range': [min(steps_low), max(steps_low)],
                        'has_temporal_data': True
                    }

        with open(output_path, 'w') as f:
            json.dump(save_dict, f)

        print(f"Saved activations to {output_path}")
        if metadata_lists is not None:
            print(f"  With temporal metadata for {len(metadata_lists.get('high', []))} high and {len(metadata_lists.get('low', []))} low examples")

    @staticmethod
    def load_activations(input_path: str) -> Tuple[Dict[int, Dict[str, List]], Dict, Optional[Dict[str, List[Dict]]]]:
        """
        Load activations from JSON file with optional temporal metadata

        Args:
            input_path: Path to activations file

        Returns:
            Tuple of (activations dict, metadata dict, temporal_metadata dict or None)
        """
        with open(input_path, 'r') as f:
            data = json.load(f)

        # Extract activations
        activations = {}
        for layer_str, acts in data['activations'].items():
            layer = int(layer_str)
            activations[layer] = {
                'high': [torch.tensor(act, dtype=torch.float32) for act in acts['high']],
                'low': [torch.tensor(act, dtype=torch.float32) for act in acts['low']]
            }

        # Extract metadata
        metadata = {
            'model_name': data.get('model_name', 'unknown'),
            'extraction_method': data.get('extraction_method', 'unknown'),
            'context': data.get('context', 'unknown'),
            'num_layers': data.get('num_layers', len(activations)),
            'num_pairs': data.get('num_pairs', 0)
        }

        # Extract temporal metadata if present
        temporal_metadata = data.get('temporal_metadata', None)

        print(f"Loaded activations from {input_path}")
        print(f"  Layers: {len(activations)}")
        print(f"  Pairs: {metadata['num_pairs']}")

        if temporal_metadata is not None:
            print(f"  With temporal metadata")
            if 'temporal_summary' in data:
                summary = data['temporal_summary']
                if 'high_step_range' in summary:
                    print(f"    High collusion steps: {summary['high_step_range'][0]}-{summary['high_step_range'][1]}")
                if 'low_step_range' in summary:
                    print(f"    Low collusion steps: {summary['low_step_range'][0]}-{summary['low_step_range'][1]}")

        return activations, metadata, temporal_metadata


if __name__ == "__main__":
    print("CAA Trainer Module")
    print("Use this module as part of the training pipeline")
