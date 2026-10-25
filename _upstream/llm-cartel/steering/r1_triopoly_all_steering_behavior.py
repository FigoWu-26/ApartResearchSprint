#!/usr/bin/env python3
"""
Improved steering tester with flexible layer selection.
Usage: python r1_duopoly_steering_tester_v2.py <prefix> <prior_steps> <size> <multiplier> <seed> [layer1] [layer2] ...

Examples:
  # Use default layer 10
  python r1_duopoly_steering_tester_v2.py test 10 7B 1.0 42

  # Use specific layer(s)
  python r1_duopoly_steering_tester_v2.py test 10 7B 1.0 42 25
  python r1_duopoly_steering_tester_v2.py test 10 7B 1.0 42 12 18
  python r1_duopoly_steering_tester_v2.py test 10 7B 1.0 42 7 11 21 25
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from environment.oligopoly import *
import sys
import os
import re
import torch
import json
from typing import Dict, List, Optional, Callable, Union, Tuple
from collections import defaultdict

# Parse command line arguments
prefix = str(sys.argv[1])
prior_steps = int(sys.argv[2])
size = str(sys.argv[3])
multiplier = float(sys.argv[4])
seed = int(sys.argv[5])

# Parse layer selection (optional)
if len(sys.argv) > 6:
    selected_layers = [int(layer) for layer in sys.argv[6:]]
    print(f"User selected layers: {selected_layers}")
else:
    # Default to layer 12 
    selected_layers = [12]
    print(f"No layers specified, using optimal layer: {selected_layers}")

# Set random seeds
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

class FlexibleSteeringApplier:
    """Apply steering vectors with flexible layer selection"""

    def __init__(self, model, tokenizer, all_steering_vectors: Dict[int, torch.Tensor],
                 selected_layers: List[int], normalize_vectors: bool = True):
        """
        Initialize with model and steering vectors for selected layers only

        Args:
            model: The language model
            tokenizer: The tokenizer
            all_steering_vectors: Dict with ALL available steering vectors
            selected_layers: List of layer indices to actually use
            normalize_vectors: Whether to normalize vectors to unit norm
        """
        self.model = model
        self.tokenizer = tokenizer
        self.hooks = []
        self.multiplier = 1.0
        self.selected_layers = selected_layers
        self.steering_mode = "all_generated"
        self.debug_mode = True
        self.activation_stats = defaultdict(list)

        # Extract only the selected layers
        self.steering_vectors = {}
        for layer in selected_layers:
            if layer in all_steering_vectors:
                vec = all_steering_vectors[layer]
                if normalize_vectors:
                    vec = vec / (vec.norm() + 1e-8)
                self.steering_vectors[layer] = vec
                print(f"✓ Loaded steering vector for layer {layer}")
            else:
                print(f"⚠ Warning: Layer {layer} not found in steering vectors file")

        if not self.steering_vectors:
            raise ValueError("No valid steering vectors loaded!")

        self._print_vector_stats()

    def _print_vector_stats(self):
        """Print statistics about loaded steering vectors"""
        print("\n" + "="*60)
        print("STEERING CONFIGURATION")
        print("="*60)
        print(f"Active layers: {sorted(self.steering_vectors.keys())}")
        print(f"Number of layers: {len(self.steering_vectors)}")
        print("\nVector Statistics:")
        for layer_idx, vec in sorted(self.steering_vectors.items()):
            print(f"  Layer {layer_idx}:")
            print(f"    Shape: {vec.shape}")
            print(f"    Norm: {vec.norm():.4f}")
            print(f"    Mean: {vec.mean():.6f}")
            print(f"    Std: {vec.std():.6f}")
            print(f"    Max abs value: {vec.abs().max():.6f}")
        print("="*60 + "\n")

    def set_steering_mode(self, mode: str):
        """Set steering application mode

        Args:
            mode: "all_generated" or "all_tokens"
        """
        valid_modes = ["all_generated", "all_tokens"]
        if mode not in valid_modes:
            raise ValueError(f"Mode must be one of {valid_modes}")
        self.steering_mode = mode
        print(f"Steering mode set to: {mode}")

    def _get_steering_hook(self, layer_idx: int, steering_vector: torch.Tensor):
        """Create a hook that adds the steering vector to activations"""

        def hook(module, input, output):
            hidden_states = output[0]

            if self.debug_mode:
                self.activation_stats[layer_idx].append({
                    'mean': hidden_states.mean().item(),
                    'std': hidden_states.std().item(),
                    'max': hidden_states.abs().max().item(),
                    'shape': hidden_states.shape
                })

            apply_steering = False
            if self.steering_mode == "all_tokens":
                apply_steering = True
            elif self.steering_mode == "all_generated":
                is_generation_step = hidden_states.shape[1] == 1
                if is_generation_step:
                    apply_steering = True

            if apply_steering:
                steering_vec = steering_vector.to(hidden_states.device).to(hidden_states.dtype)

                if hidden_states.dim() == 3:  # Shape is [batch, seq_len, hidden_dim]
                    additive_vector = self.multiplier * steering_vec.unsqueeze(0).unsqueeze(0)
                    hidden_states = hidden_states + additive_vector

            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            else:
                return hidden_states

        return hook

    def activate_steering(self):
        """Activate steering by registering hooks"""
        self.deactivate_steering()

        if self.debug_mode:
            print(f"\nActivating steering:")
            print(f"  Mode: {self.steering_mode}")
            print(f"  Layers: {sorted(self.steering_vectors.keys())}")
            print(f"  Multiplier: {self.multiplier}")

        for layer_idx, steering_vec in self.steering_vectors.items():
            hook = self.model.model.layers[layer_idx].register_forward_hook(
                self._get_steering_hook(layer_idx, steering_vec)
            )
            self.hooks.append(hook)

    def deactivate_steering(self):
        """Remove all steering hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activation_stats.clear()

    def set_multiplier(self, multiplier: float):
        """Set the steering strength multiplier"""
        self.multiplier = multiplier
        if self.debug_mode:
            print(f"Steering multiplier set to {self.multiplier}")

    def generate_with_steering(self,
                               prompt: Union[str, List[Dict], List[List[Dict]]],
                               multiplier: float = 1.0,
                               max_new_tokens: int = 300,
                               temperature: float = 0.7,
                               debug: bool = False,
                               **kwargs) -> List[str]:
        """Generate text with steering applied - handles batch inputs like v1"""
        self.set_multiplier(multiplier)
        self.debug_mode = debug

        # Replicate v1 behavior exactly
        if isinstance(prompt, str):
            prompt_text = prompt
        elif isinstance(prompt, list):
            # This handles both single chat history and batch of chat histories
            # The tokenizer.apply_chat_template handles both cases
            prompt_text = self.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
        else:
            raise ValueError("Prompt must be a string or a list of message dictionaries.")

        # Tokenize - handles both single strings and lists of strings
        try:
            inputs = self.tokenizer(prompt_text, return_tensors="pt", padding=True)
        except:
            print("failure to tokenize!!!")
            print("#######################")
            print(prompt_text)
            print("#######################")
            raise

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        self.prompt_length = inputs['input_ids'].shape[1]

        if debug:
            print(f"\nGenerating with steering:")
            print(f"  Prompt length: {self.prompt_length} tokens")
            print(f"  Batch size: {inputs['input_ids'].shape[0]}")
            print(f"  Steering mode: {self.steering_mode}")

        self.activate_steering()

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **kwargs
                )

            generated_list = []

            for ind in range(len(outputs)):
                generated_tokens = outputs[ind][self.prompt_length:]
                generated = self.tokenizer.decode(
                    generated_tokens, skip_special_tokens=False
                )

                # Clean up special tokens like in v1
                full_text = generated.replace("<｜end▁of▁sentence｜>","")
                generated_list.append(full_text)

            if debug and self.activation_stats:
                self._print_activation_stats()

            return generated_list

        finally:
            self.deactivate_steering()

    def _print_activation_stats(self):
        """Print activation statistics from debug mode"""
        print("\nActivation Statistics:")
        for layer_idx, stats_list in self.activation_stats.items():
            if stats_list:
                recent_stats = stats_list[-5:]
                print(f"  Layer {layer_idx} (last {len(recent_stats)} forward passes):")
                for i, stats in enumerate(recent_stats):
                    print(f"    Pass {i}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                          f"max={stats['max']:.4f}, shape={stats['shape']}")


def load_all_steering_vectors(path: str) -> Dict[int, torch.Tensor]:
    """Load ALL steering vectors from JSON file"""
    print(f"\nLoading steering vectors from: {path}")

    with open(path, 'r') as f:
        save_dict = json.load(f)

    steering_vectors = {
        int(k): torch.tensor(v, dtype=torch.float32)
        for k, v in save_dict['vectors'].items()
    }

    print(f"Total layers available: {sorted(steering_vectors.keys())}")
    print(f"Model: {save_dict.get('model_name', 'Unknown')}")
    print(f"Method: {save_dict.get('extraction_method', 'Unknown')}")

    return steering_vectors


# Initialize model
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-" + size
print(f"\nLoading model: {model_name}")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device (Apple Silicon)")
else:
    device = torch.device("cpu")
    print("Using CPU device")

if device.type == "cuda":
    dtype = torch.bfloat16
elif device.type == "mps":
    dtype = torch.float16
else:
    dtype = torch.float32

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=dtype,
    device_map="auto" if device.type == "cuda" else None,
    trust_remote_code=True
)

if device.type == "mps":
    model = model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load ALL available steering vectors
vector_path = "results/steering_vectors/all_layers_caa_vectors_behavior.json"
if not os.path.exists(vector_path):
    print(f"ERROR: Steering vector file not found: {vector_path}")
    print("Please run: python steering/compute_all_layer_vectors.py first")
    sys.exit(1)
all_steering_vectors = load_all_steering_vectors(vector_path)

# Create steering applier with only selected layers
llm = FlexibleSteeringApplier(
    model,
    tokenizer,
    all_steering_vectors,
    selected_layers=selected_layers,
    normalize_vectors=True
)

# Rest of the game logic remains the same...

def extract_price(response_text):
    """
    Extract price from response text, handling various formats.
    Returns the price as float, or None if not found.
    """
    # Remove the think tags if present
    if "</think>" in response_text:
        response_text = response_text.split("</think>")[-1]

    # Look for price in various formats
    for line in response_text.split("\n"):
        line_lower = line.lower()

        # Check for "chosen price:" or "my chosen price:"
        if "chosen price:" in line_lower:
            # Extract everything after "price:"
            price_part = line.split(":")[-1].strip()

            # Remove common prefixes/suffixes: $, USD, dollars, etc.
            price_part = price_part.replace("$", "").replace("USD", "")
            price_part = price_part.replace("dollars", "").replace("dollar", "")

            # Extract the first number found
            numbers = re.findall(r'\d+\.?\d*', price_part)
            if numbers:
                return round(float(numbers[0]), 2)

    return None


def getActions(prompts,old_prices):
    """Get actions from multiple agents in batch."""
    actions = []
    thoughts = []
    responses = []

    # Generate responses for all agents at once
    answer = llm.generate_with_steering(prompts, multiplier=multiplier,
                                       max_new_tokens=4096, temperature=1.0)

    print("answers=", answer)
    print("answer_len=", len(answer))

    # Process each agent's response
    for i in range(len(answer)):
        all_content = answer[i]

        # Split thought and response
        if "</think>" in all_content:
            parts = all_content.split("</think>")
            thought = parts[0]
            response = parts[-1]
        else:
            thought = ""
            response = all_content

        # Try to extract price
        price = extract_price(response)

        if price is not None:
            actions.append(price)
            thoughts.append(thought)
            responses.append(response)
        else:
            # Retry with individual generation
            print(f"Parse failed for agent {i}, retrying...")
            retry_succeeded = False
            for retry in range(10):
                try:
                    price, thought, response = getAction(prompts[i])
                    if price is not None:
                        actions.append(price)
                        thoughts.append(thought)
                        responses.append(response)
                        retry_succeeded = True
                        break
                except:
                    print(f"Retry {retry + 1} failed for agent {i}")

            if not retry_succeeded:
                print(f"All retries failed for agent {i}")
                # Don't add anything - let the failure handling below take care of it

    # Handle failure cases like in original code
    if len(actions) == 0:
        print("TOTAL FAILURE::: DEFAULT TO LAST ACTIONS")
        actions = old_prices
        thoughts = []
        responses = []
    elif len(actions) < len(prompts):
        print("PARTIAL FAILURE::: DEFAULT TO SAME ACTIONS")
        while len(actions) < len(prompts):
            actions.append(actions[0])
            thoughts.append(thoughts[0] if thoughts else "")
            responses.append(responses[0] if responses else "")

    return actions, thoughts, responses

def getAction(prompt):
    """Get action from a single agent."""
    answer = llm.generate_with_steering([prompt], multiplier=multiplier,
                                       max_new_tokens=8192, temperature=1.0)

    # generate_with_steering always returns a list
    all_content = answer[0] if isinstance(answer, list) else answer

    # Split thought and response
    if "</think>" in all_content:
        parts = all_content.split("</think>")
        thought = parts[0]
        response = parts[-1]
    else:
        thought = ""
        response = all_content

    # Extract price
    price = extract_price(response)

    if price is None:
        raise ValueError("Could not extract price from response")

    return price, thought, response

selected_layers_str = str(selected_layers[0])
for episode in range(10):
    print("episode=", episode)
    env = OligopolyGame(a0=0, ais=[3, 3, 3], costs=[1, 1, 1], mu=0.315, alpha=1.0, prefix=prefix)
    prompts = env.reset(prior_steps=prior_steps)
    prices = [0.0, 0.0]
    for step in range(300):
        print("step=", step)
        actions, thoughts, responses = getActions(prompts,prices)
        filename = "results/jsonl/beh_triopoly_steering_caa"+str(multiplier)+"_"+str(prior_steps)+"_prefix_"+str(prefix)+"_size_"+str(size)+"_ep_"+str(episode)+"_layer_"+selected_layers_str+"_seed_"+str(seed)+".jsonl"

        prices, demands, profits, gains, prompts = env.step(actions)
        print("episode=", episode, "round=", step, "prices=", actions, "demands=",demands,"profits=",profits,"gains=",gains)
        sys.stdout.flush()

        with open(filename, "a") as file:
            new_output = {"round": step, "actions": actions, "thoughts": thoughts, "respones": responses, "demands": demands,"profits": profits,"gains": gains}
            file.write(json.dumps(new_output) + "\n")





