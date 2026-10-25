"""
Data Loader Module

Loads and processes collusion-labeled datasets for steering vector training.
Supports JSONL format with high/low collusion examples.
Enhanced with temporal metadata tracking for turn-based analysis.
"""

import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
import warnings


class DataLoader:
    """Load and process collusion-labeled data for steering vector training"""

    def __init__(self, high_collusion_path: str, low_collusion_path: str):
        """
        Initialize the data loader with paths to high and low collusion datasets

        Args:
            high_collusion_path: Path to JSONL file with high collusion examples
            low_collusion_path: Path to JSONL file with low collusion examples
        """
        self.high_collusion_path = Path(high_collusion_path)
        self.low_collusion_path = Path(low_collusion_path)
        self.high_data: List[Dict] = []
        self.low_data: List[Dict] = []

        # Validate paths
        if not self.high_collusion_path.exists():
            raise FileNotFoundError(f"High collusion file not found: {high_collusion_path}")
        if not self.low_collusion_path.exists():
            raise FileNotFoundError(f"Low collusion file not found: {low_collusion_path}")

    def _load_jsonl_file(self, file_path: Path) -> List[Dict]:
        """
        Load data from a single JSONL file

        Args:
            file_path: Path to JSONL file

        Returns:
            List of dictionaries containing example data
        """
        data_list = []
        print(f"Loading data from {file_path.name}")

        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)

                    # Validate required fields
                    if 'thoughts' not in data:
                        warnings.warn(f"Line {line_num}: Missing 'thoughts' field, skipping")
                        continue
                    if 'prompt' not in data:
                        warnings.warn(f"Line {line_num}: Missing 'prompt' field, skipping")
                        continue
                    if 'collusion_score' not in data:
                        warnings.warn(f"Line {line_num}: Missing 'collusion_score' field, skipping")
                        continue

                    data_list.append(data)

                except json.JSONDecodeError as e:
                    warnings.warn(f"Line {line_num}: Invalid JSON - {e}, skipping")
                except Exception as e:
                    warnings.warn(f"Line {line_num}: Error processing line - {e}, skipping")

        return data_list

    def load_data(self, verbose: bool = True) -> None:
        """
        Load data from both high and low collusion files

        Args:
            verbose: Whether to print loading statistics
        """
        self.high_data = self._load_jsonl_file(self.high_collusion_path)
        self.low_data = self._load_jsonl_file(self.low_collusion_path)

        if verbose:
            print(f"\nData loading summary:")
            print(f"  High collusion examples: {len(self.high_data)}")
            print(f"  Low collusion examples: {len(self.low_data)}")

            if self.high_data:
                high_scores = [d['collusion_score'] for d in self.high_data]
                print(f"  High collusion score range: {min(high_scores):.3f} - {max(high_scores):.3f}")

            if self.low_data:
                low_scores = [d['collusion_score'] for d in self.low_data]
                print(f"  Low collusion score range: {min(low_scores):.3f} - {max(low_scores):.3f}")

    def filter_by_step_range(self, data: List[Dict],
                            step_range: Optional[Tuple[int, int]] = None) -> List[Dict]:
        """
        Filter data by step/turn range

        Args:
            data: List of data dictionaries
            step_range: Optional tuple of (min_step, max_step), inclusive

        Returns:
            Filtered list of data
        """
        if step_range is None:
            return data

        min_step, max_step = step_range
        filtered = []

        for item in data:
            if 'step' in item:
                step = item['step']
                if min_step <= step <= max_step:
                    filtered.append(item)
            else:
                # If no step field, include the item (backward compatibility)
                filtered.append(item)

        return filtered

    def get_temporal_metadata(self, data_item: Dict) -> Dict[str, Any]:
        """
        Extract temporal metadata from a data item

        Args:
            data_item: Single data dictionary

        Returns:
            Dictionary with temporal metadata (step, episode_id if available)
        """
        metadata = {}

        # Extract step/turn number
        if 'step' in data_item:
            metadata['step'] = data_item['step']
        else:
            metadata['step'] = None

        # Extract episode/seed if available (for future use)
        if 'episode_id' in data_item:
            metadata['episode_id'] = data_item['episode_id']
        elif 'seed' in data_item:
            metadata['seed'] = data_item['seed']

        # Extract collusion score for analysis
        if 'collusion_score' in data_item:
            metadata['collusion_score'] = data_item['collusion_score']

        return metadata

    def get_prompt_thought_pairs(self,
                                max_pairs: Optional[int] = None,
                                balance: bool = True,
                                step_range: Optional[Tuple[int, int]] = None) -> List[Tuple[List[Dict], str, List[Dict], str]]:
        """
        Extract pairs of (prompt, thought) for high and low collusion examples

        Args:
            max_pairs: Maximum number of pairs to return (None = all)
            balance: Whether to balance high/low examples (take min of both)
            step_range: Optional tuple of (min_step, max_step) to filter by turn number

        Returns:
            List of (high_prompt, high_thought, low_prompt, low_thought) tuples
        """
        # Apply step range filtering if specified
        high_data = self.filter_by_step_range(self.high_data, step_range)
        low_data = self.filter_by_step_range(self.low_data, step_range)

        if step_range is not None:
            print(f"Filtering to step range {step_range[0]}-{step_range[1]}: {len(high_data)} high, {len(low_data)} low examples")

        pairs = []

        # Determine number of pairs to create
        if balance:
            num_pairs = min(len(high_data), len(low_data))
        else:
            num_pairs = max(len(high_data), len(low_data))

        if max_pairs is not None:
            num_pairs = min(num_pairs, max_pairs)

        for i in range(num_pairs):
            # Handle cases where one list is shorter
            high_idx = i % len(high_data)
            low_idx = i % len(low_data)

            high_prompt = high_data[high_idx]['prompt']
            high_thought = high_data[high_idx]['thoughts']
            low_prompt = low_data[low_idx]['prompt']
            low_thought = low_data[low_idx]['thoughts']

            # Validate that prompts are lists of dicts with role/content
            if not (isinstance(high_prompt, list) and isinstance(low_prompt, list)):
                warnings.warn(f"Pair {i}: Invalid prompt format, skipping")
                continue

            # Validate prompt structure
            valid_high = all(isinstance(m, dict) and 'role' in m and 'content' in m
                           for m in high_prompt)
            valid_low = all(isinstance(m, dict) and 'role' in m and 'content' in m
                          for m in low_prompt)

            if not (valid_high and valid_low):
                warnings.warn(f"Pair {i}: Invalid message structure, skipping")
                continue

            pairs.append((high_prompt, high_thought, low_prompt, low_thought))

        print(f"\nCreated {len(pairs)} prompt-thought pairs")
        return pairs

    def get_prompt_thought_pairs_with_metadata(self,
                                              max_pairs: Optional[int] = None,
                                              balance: bool = True,
                                              step_range: Optional[Tuple[int, int]] = None) -> List[Dict]:
        """
        Extract pairs with temporal metadata for advanced analysis

        Args:
            max_pairs: Maximum number of pairs to return (None = all)
            balance: Whether to balance high/low examples (take min of both)
            step_range: Optional tuple of (min_step, max_step) to filter by turn number

        Returns:
            List of dictionaries containing:
                - high_prompt, high_thought, low_prompt, low_thought
                - high_metadata, low_metadata (temporal info)
        """
        # Apply step range filtering if specified
        high_data = self.filter_by_step_range(self.high_data, step_range)
        low_data = self.filter_by_step_range(self.low_data, step_range)

        pairs_with_metadata = []

        # Determine number of pairs to create
        if balance:
            num_pairs = min(len(high_data), len(low_data))
        else:
            num_pairs = max(len(high_data), len(low_data))

        if max_pairs is not None:
            num_pairs = min(num_pairs, max_pairs)

        for i in range(num_pairs):
            # Handle cases where one list is shorter
            high_idx = i % len(high_data)
            low_idx = i % len(low_data)

            high_item = high_data[high_idx]
            low_item = low_data[low_idx]

            # Extract prompts and thoughts
            high_prompt = high_item['prompt']
            high_thought = high_item['thoughts']
            low_prompt = low_item['prompt']
            low_thought = low_item['thoughts']

            # Validate prompts
            if not (isinstance(high_prompt, list) and isinstance(low_prompt, list)):
                continue

            valid_high = all(isinstance(m, dict) and 'role' in m and 'content' in m
                           for m in high_prompt)
            valid_low = all(isinstance(m, dict) and 'role' in m and 'content' in m
                          for m in low_prompt)

            if not (valid_high and valid_low):
                continue

            # Extract metadata
            high_metadata = self.get_temporal_metadata(high_item)
            low_metadata = self.get_temporal_metadata(low_item)

            pair_data = {
                'high_prompt': high_prompt,
                'high_thought': high_thought,
                'low_prompt': low_prompt,
                'low_thought': low_thought,
                'high_metadata': high_metadata,
                'low_metadata': low_metadata
            }

            pairs_with_metadata.append(pair_data)

        print(f"\nCreated {len(pairs_with_metadata)} prompt-thought pairs with metadata")
        return pairs_with_metadata

    def get_statistics(self) -> Dict:
        """
        Get statistics about the loaded data

        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'num_high': len(self.high_data),
            'num_low': len(self.low_data),
        }

        if self.high_data:
            high_scores = [d['collusion_score'] for d in self.high_data]
            stats['high_score_mean'] = sum(high_scores) / len(high_scores)
            stats['high_score_min'] = min(high_scores)
            stats['high_score_max'] = max(high_scores)

        if self.low_data:
            low_scores = [d['collusion_score'] for d in self.low_data]
            stats['low_score_mean'] = sum(low_scores) / len(low_scores)
            stats['low_score_min'] = min(low_scores)
            stats['low_score_max'] = max(low_scores)

        return stats

    @staticmethod
    def list_available_datasets(datasets_dir: str = "results/datasets") -> Dict[str, Dict[str, str]]:
        """
        List all available dataset pairs in the datasets directory

        Args:
            datasets_dir: Directory to search for datasets

        Returns:
            Dictionary mapping dataset names to their high/low paths
        """
        datasets_path = Path(datasets_dir)
        if not datasets_path.exists():
            print(f"Datasets directory not found: {datasets_dir}")
            return {}

        # Find all JSONL files
        all_files = list(datasets_path.glob("*.jsonl"))

        # Group by dataset name (everything before _high or _low)
        datasets = {}
        for file in all_files:
            name = file.stem

            # Determine if this is high or low
            if '_high_scores' in name:
                base_name = name.replace('_high_scores', '')
                if base_name not in datasets:
                    datasets[base_name] = {}
                datasets[base_name]['high'] = str(file)
            elif '_low_scores' in name:
                base_name = name.replace('_low_scores', '')
                if base_name not in datasets:
                    datasets[base_name] = {}
                datasets[base_name]['low'] = str(file)

        # Only keep datasets that have both high and low
        complete_datasets = {
            name: paths for name, paths in datasets.items()
            if 'high' in paths and 'low' in paths
        }

        return complete_datasets


def test_data_loader():
    """Test the data loader functionality"""
    print("Testing DataLoader...")

    # List available datasets
    print("\nAvailable datasets:")
    datasets = DataLoader.list_available_datasets()
    for name, paths in datasets.items():
        print(f"  {name}:")
        print(f"    High: {Path(paths['high']).name}")
        print(f"    Low: {Path(paths['low']).name}")

    # Test loading a specific dataset
    if datasets:
        first_dataset = list(datasets.keys())[0]
        paths = datasets[first_dataset]

        print(f"\nTesting with dataset: {first_dataset}")
        loader = DataLoader(paths['high'], paths['low'])
        loader.load_data()

        # Get pairs
        pairs = loader.get_prompt_thought_pairs(max_pairs=5)
        print(f"\nSample pair (first):")
        if pairs:
            high_prompt, high_thought, low_prompt, low_thought = pairs[0]
            print(f"  High prompt messages: {len(high_prompt)}")
            print(f"  High thought preview: {high_thought[:100]}...")
            print(f"  Low thought preview: {low_thought[:100]}...")

        # Get statistics
        stats = loader.get_statistics()
        print(f"\nDataset statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    test_data_loader()
