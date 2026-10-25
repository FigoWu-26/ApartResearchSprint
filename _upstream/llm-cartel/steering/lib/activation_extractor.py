"""
Activation Extractor Module

Extracts hidden state activations from transformer models for steering vector training.
Supports extraction from all layers, using thought tokens only with full prompt context.
Enhanced with temporal metadata tracking for turn-based analysis.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Dict, Optional, Any
from tqdm import tqdm
import warnings


class ActivationExtractor:
    """Extract activations from language models for steering vector construction"""

    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"):
        """
        Initialize with a transformer model

        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name

        # Device handling for CUDA, MPS (Apple Silicon), or CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.dtype = torch.bfloat16  # CUDA supports bfloat16
            print(f"Using CUDA device")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.dtype = torch.float16  # MPS only supports float16
            print(f"Using MPS device (Apple Silicon)")
        else:
            self.device = torch.device("cpu")
            self.dtype = torch.float32
            print(f"Warning: Using CPU device - this will be slow")

        print(f"Loading model {model_name}...")

        # Load model with appropriate dtype
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map="auto" if self.device.type == "cuda" else None,
            trust_remote_code=True
        )

        # For MPS, manually move to device after loading
        if self.device.type == "mps":
            self.model = self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Get model configuration
        self.num_layers = self.model.config.num_hidden_layers
        self.hidden_dim = self.model.config.hidden_size

        print(f"Model loaded: {self.num_layers} layers, {self.hidden_dim} hidden dimensions")

        # Hook storage for capturing activations
        self.activations: Dict[int, torch.Tensor] = {}

    def _get_activation_hook(self, layer_idx: int):
        """
        Create a hook function to capture activations at a specific layer

        Args:
            layer_idx: Index of the layer to hook

        Returns:
            Hook function
        """
        def hook(module, input, output):
            # output[0] contains the hidden states (residual stream after layer)
            # Shape: [batch_size, sequence_length, hidden_dim]
            self.activations[layer_idx] = output[0].detach().cpu()

        return hook

    def format_prompt_with_thought(self, prompt: List[Dict], thought: str) -> str:
        """
        Format prompt and thought into proper model format

        Args:
            prompt: List of message dictionaries with role/content
            thought: The thinking text to add

        Returns:
            Formatted text ready for tokenization
        """
        # Apply chat template to the prompt
        prompt_text = self.tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True
        )

        # For R1 models, add thought as assistant response with think tags
        # The prompt already ends with the assistant marker from apply_chat_template
        full_text = prompt_text + f"<think>{thought}</think>"

        return full_text

    def extract_thought_activations(self,
                                   prompt: List[Dict],
                                   thought: str,
                                   layer_indices: Optional[List[int]] = None) -> Dict[int, torch.Tensor]:
        """
        Extract activations only from thought tokens, using full context for inference

        Args:
            prompt: Conversation prompt (list of role/content dicts)
            thought: The thinking text
            layer_indices: List of layer indices to extract from (None = all layers)

        Returns:
            Dictionary mapping layer index to activation tensor (averaged over thought tokens)
        """
        # Clear previous activations
        self.activations = {}

        # Default to all layers if not specified
        if layer_indices is None:
            layer_indices = list(range(self.num_layers))

        # Validate layer indices
        invalid_layers = [l for l in layer_indices if l < 0 or l >= self.num_layers]
        if invalid_layers:
            raise ValueError(f"Invalid layer indices: {invalid_layers}. "
                           f"Model has {self.num_layers} layers (0-{self.num_layers-1})")

        # Format the full sequence (prompt + thought)
        full_text = self.format_prompt_with_thought(prompt, thought)

        # Tokenize full sequence
        full_inputs = self.tokenizer(full_text, return_tensors="pt")
        full_inputs = {k: v.to(self.device) for k, v in full_inputs.items()}

        # Also tokenize just the prompt to find where thoughts start
        prompt_only_text = self.tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True
        )
        prompt_only_inputs = self.tokenizer(prompt_only_text, return_tensors="pt")

        # Find where thought tokens start
        prompt_token_count = prompt_only_inputs['input_ids'].shape[1]
        full_token_count = full_inputs['input_ids'].shape[1]
        thought_token_count = full_token_count - prompt_token_count

        if thought_token_count <= 0:
            warnings.warn("No thought tokens found - thought might be empty")
            thought_token_count = 1  # Use at least one token

        # Register hooks for specified layers
        hooks = []
        for layer_idx in layer_indices:
            hook = self.model.model.layers[layer_idx].register_forward_hook(
                self._get_activation_hook(layer_idx)
            )
            hooks.append(hook)

        # Forward pass with full context (no gradients needed)
        try:
            with torch.no_grad():
                _ = self.model(**full_inputs)
        finally:
            # Always remove hooks, even if forward pass fails
            for hook in hooks:
                hook.remove()

        # Extract activations only from thought tokens
        extracted = {}
        for layer_idx in layer_indices:
            if layer_idx not in self.activations:
                warnings.warn(f"Layer {layer_idx}: No activations captured")
                continue

            acts = self.activations[layer_idx]
            # acts shape: [batch, seq_len, hidden_dim]

            # Extract only the thought token activations
            thought_acts = acts[:, prompt_token_count:full_token_count, :]

            # Average over thought tokens to get single vector per example
            avg_thought_acts = thought_acts.mean(dim=1).squeeze()

            # Ensure we have the right shape
            if avg_thought_acts.dim() == 0:
                avg_thought_acts = avg_thought_acts.unsqueeze(0)

            # CRITICAL: Convert to float32 to avoid precision loss!
            # Float16 can cause overflow/underflow in downstream computations
            avg_thought_acts = avg_thought_acts.to(torch.float32)

            extracted[layer_idx] = avg_thought_acts

        # Clear GPU/MPS cache
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return extracted

    def extract_from_prompt_thought_pairs(self,
                                         prompt_thought_pairs: List[Tuple[List[Dict], str, List[Dict], str]],
                                         layer_indices: Optional[List[int]] = None,
                                         show_progress: bool = True) -> Dict[int, Dict[str, List[torch.Tensor]]]:
        """
        Extract activations from a list of prompt-thought pairs

        Args:
            prompt_thought_pairs: List of (high_prompt, high_thought, low_prompt, low_thought) tuples
            layer_indices: Layers to extract from (None = all layers)
            show_progress: Whether to show progress bar

        Returns:
            Dictionary with structure: {layer_idx: {'high': [act1, ...], 'low': [act1, ...]}}
        """
        # Default to all layers if not specified
        if layer_indices is None:
            layer_indices = list(range(self.num_layers))

        print(f"\nExtracting activations for {len(prompt_thought_pairs)} pairs from {len(layer_indices)} layers...")

        # Initialize storage
        layer_activations = {
            layer: {'high': [], 'low': []}
            for layer in layer_indices
        }

        # Process each pair with progress bar
        iterator = tqdm(prompt_thought_pairs, desc="Processing pairs") if show_progress else prompt_thought_pairs

        for high_prompt, high_thought, low_prompt, low_thought in iterator:
            # Extract high collusion activations
            try:
                high_acts = self.extract_thought_activations(high_prompt, high_thought, layer_indices)

                # Store activations for each layer
                for layer in layer_indices:
                    if layer in high_acts:
                        layer_activations[layer]['high'].append(high_acts[layer])

            except Exception as e:
                warnings.warn(f"Error extracting high activations: {e}")
                continue

            # Extract low collusion activations
            try:
                low_acts = self.extract_thought_activations(low_prompt, low_thought, layer_indices)

                # Store activations for each layer
                for layer in layer_indices:
                    if layer in low_acts:
                        layer_activations[layer]['low'].append(low_acts[layer])

            except Exception as e:
                warnings.warn(f"Error extracting low activations: {e}")
                # Remove corresponding high activation if low failed
                for layer in layer_indices:
                    if layer_activations[layer]['high']:
                        layer_activations[layer]['high'].pop()
                continue

            # Periodic cache clearing (every 10 pairs)
            if len(layer_activations[layer_indices[0]]['high']) % 10 == 0:
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        # Print summary
        print(f"\nExtraction complete:")
        for layer in layer_indices[:3]:  # Show first 3 layers
            n_high = len(layer_activations[layer]['high'])
            n_low = len(layer_activations[layer]['low'])
            print(f"  Layer {layer}: {n_high} high examples, {n_low} low examples")

        if len(layer_indices) > 3:
            print(f"  ... and {len(layer_indices) - 3} more layers")

        return layer_activations

    def extract_from_pairs_with_metadata(self,
                                        pairs_with_metadata: List[Dict],
                                        layer_indices: Optional[List[int]] = None,
                                        show_progress: bool = True) -> Tuple[Dict[int, Dict[str, List[torch.Tensor]]], Dict[str, List[Dict]]]:
        """
        Extract activations from pairs that include metadata

        Args:
            pairs_with_metadata: List of dictionaries with keys:
                - high_prompt, high_thought, low_prompt, low_thought
                - high_metadata, low_metadata
            layer_indices: Layers to extract from (None = all layers)
            show_progress: Whether to show progress bar

        Returns:
            Tuple of:
                - Dictionary with structure: {layer_idx: {'high': [act1, ...], 'low': [act1, ...]}}
                - Dictionary with metadata: {'high': [meta1, ...], 'low': [meta1, ...]}
        """
        # Default to all layers if not specified
        if layer_indices is None:
            layer_indices = list(range(self.num_layers))

        print(f"\nExtracting activations with metadata for {len(pairs_with_metadata)} pairs from {len(layer_indices)} layers...")

        # Initialize storage for activations and metadata
        layer_activations = {
            layer: {'high': [], 'low': []}
            for layer in layer_indices
        }

        metadata_lists = {'high': [], 'low': []}

        # Process each pair with progress bar
        iterator = tqdm(pairs_with_metadata, desc="Processing pairs") if show_progress else pairs_with_metadata

        for pair_data in iterator:
            # Extract components
            high_prompt = pair_data['high_prompt']
            high_thought = pair_data['high_thought']
            low_prompt = pair_data['low_prompt']
            low_thought = pair_data['low_thought']
            high_metadata = pair_data.get('high_metadata', {})
            low_metadata = pair_data.get('low_metadata', {})

            # Extract high collusion activations
            try:
                high_acts = self.extract_thought_activations(high_prompt, high_thought, layer_indices)

                # Store activations and metadata for each layer
                for layer in layer_indices:
                    if layer in high_acts:
                        layer_activations[layer]['high'].append(high_acts[layer])

                # Store metadata
                metadata_lists['high'].append(high_metadata)

            except Exception as e:
                warnings.warn(f"Error extracting high activations: {e}")
                continue

            # Extract low collusion activations
            try:
                low_acts = self.extract_thought_activations(low_prompt, low_thought, layer_indices)

                # Store activations and metadata for each layer
                for layer in layer_indices:
                    if layer in low_acts:
                        layer_activations[layer]['low'].append(low_acts[layer])

                # Store metadata
                metadata_lists['low'].append(low_metadata)

            except Exception as e:
                warnings.warn(f"Error extracting low activations: {e}")
                # Remove corresponding high activation and metadata if low failed
                for layer in layer_indices:
                    if layer_activations[layer]['high']:
                        layer_activations[layer]['high'].pop()
                if metadata_lists['high']:
                    metadata_lists['high'].pop()
                continue

            # Periodic cache clearing (every 10 pairs)
            if len(layer_activations[layer_indices[0]]['high']) % 10 == 0:
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        # Print summary
        print(f"\nExtraction complete:")
        for layer in layer_indices[:3]:  # Show first 3 layers
            n_high = len(layer_activations[layer]['high'])
            n_low = len(layer_activations[layer]['low'])
            print(f"  Layer {layer}: {n_high} high examples, {n_low} low examples")

        if len(layer_indices) > 3:
            print(f"  ... and {len(layer_indices) - 3} more layers")

        # Print temporal metadata summary if available
        if metadata_lists['high'] and 'step' in metadata_lists['high'][0]:
            steps_high = [m['step'] for m in metadata_lists['high'] if m.get('step') is not None]
            steps_low = [m['step'] for m in metadata_lists['low'] if m.get('step') is not None]
            if steps_high:
                print(f"\nTemporal coverage:")
                print(f"  High examples: steps {min(steps_high)}-{max(steps_high)}")
                print(f"  Low examples: steps {min(steps_low)}-{max(steps_low)}")

        return layer_activations, metadata_lists

    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model

        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'num_layers': self.num_layers,
            'hidden_dim': self.hidden_dim,
            'device': str(self.device),
            'dtype': str(self.dtype),
        }


def test_activation_extractor():
    """Test the activation extractor functionality"""
    print("Testing ActivationExtractor...")

    # Initialize extractor
    extractor = ActivationExtractor()

    # Print model info
    info = extractor.get_model_info()
    print("\nModel information:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Test with a simple example
    test_prompt = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What price should I set in a duopoly market?"}
    ]
    test_thought = "I need to consider competitor behavior and market dynamics. Setting prices too high risks losing customers."

    # Extract from a few layers
    test_layers = [0, extractor.num_layers // 2, extractor.num_layers - 1]
    print(f"\nTesting extraction from layers: {test_layers}")

    activations = extractor.extract_thought_activations(
        test_prompt, test_thought, test_layers
    )

    print("\nExtracted activations:")
    for layer, acts in activations.items():
        print(f"  Layer {layer}: shape={acts.shape}, mean={acts.mean():.4f}, std={acts.std():.4f}")


if __name__ == "__main__":
    test_activation_extractor()
