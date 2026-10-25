import json
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import re
import textwrap
from collections import Counter

mpl.rcParams["figure.max_open_warning"] = 10


# Plot function
def plot_block_frequencies(data, dataset, model_name, problem_tag):
    jsonl_results = []
    for block in data:
        responses = []
        for r in block["responses"]:
            if r is None:
                continue

            # Use last occurrence of <answer>...</answer>, even if earlier tags are malformed/nested.
            last_open = r.rfind("<answer>")
            if last_open != -1:
                last_close = r.find("</answer>", last_open)
                if last_close != -1 and last_close > last_open:
                    cleaned_response = r[last_open : last_close + len("</answer>")]
                else:
                    cleaned_response = f"<answer>{r}</answer>"
            else:
                cleaned_response = f"<answer>{r}</answer>"

            responses.append(cleaned_response)
        count = Counter(responses)
        print("\n".join(f"{k}: {v}" for k, v in count.items()))

        if count:
            # Coordination Index
            total = sum(count.values())
            if total > 1:
                coordination_index, normalised_coordination_index = (
                    compute_coordination_index(count, block["normalization_factor"])
                )
                print(f"Coordination Index: {coordination_index}")
                print(f"Normalised Coordination Index: {normalised_coordination_index}")
            else:
                coordination_index = 0.0

            # Create the figure with two subplots side by side
            fig, (ax_bar, ax_text) = plt.subplots(
                1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2, 1]}
            )

            # Plot the bar chart on the left (truncate long labels to avoid bitmap issues)
            x_positions = range(len(count))
            truncated_labels = [
                (k[:60] + "…") if len(k) > 60 else k for k in count.keys()
            ]
            ax_bar.bar(x_positions, count.values())
            ax_bar.set_xticks(list(x_positions))
            ax_bar.set_xticklabels(truncated_labels, rotation=45, ha="right")
            ax_bar.set_title(
                f"Frequencies for prompt with idx {block['idx']}.{block['variation-idx']}"
            )
            ax_bar.set_xlabel("Answer")
            ax_bar.set_ylabel("Frequency")

            # Add the wrapped prompt text on the right
            max_line_width = 45  # characters
            wrapped_text = "\n".join(
                textwrap.wrap(block["prompt"], width=max_line_width)
            )
            ax_text.text(
                0,
                1,
                wrapped_text,
                fontsize=12,
                va="top",
                bbox=dict(
                    facecolor="lightblue", edgecolor="black", boxstyle="round,pad=0.5"
                ),
            )
            ax_text.axis("off")  # Hide axis for the text box

            plt.tight_layout()
            print(block["idx"])

            # Save the figure
            plt.savefig(
                f"./images/{model_name}/{dataset}/{problem_tag}/idx_{block['idx']}.{block['variation-idx']}_frequencies.png",
                bbox_inches="tight",
                dpi=150,
            )
            plt.close(fig)

            jsonl_results.append(
                {
                    "idx": block["idx"],
                    "variation-idx": block["variation-idx"],
                    "prompt": block["prompt"],
                    "responses": dict(count),
                    "coordination_index": coordination_index,
                    "normalised_coordination_index": normalised_coordination_index,
                }
            )

        else:
            print(
                f"No valid responses found for block with idx {block['idx']} and prompt `{block['prompt']}'."
            )

        print("=" * 80)

    with open(f"./results/{model_name}/{dataset}_{problem_tag}.jsonl", "w") as f:
        json.dump(jsonl_results, f, indent=2)


def iterate_data(data: dict, problem_tag: str):
    problems, normalization_factors = {}, {}
    for d in data:
        idx = d["id"]
        problems[idx] = []
        text = d[problem_tag]
        placeholders = [d[p] for p in d["placeholders"]]
        if not placeholders:
            problems[idx].append(text)
            continue

        for elements in itertools.product(*placeholders):
            new_text = text
            assert len(elements) == len(
                d["placeholders"]
            ), "Mismatch in number of placeholders and elements"
            for el, p in zip(elements, d["placeholders"]):
                new_text = new_text.replace(f"@{p}@", el)
            problems[idx].append(new_text)

        if d["normalization_factor"]:
            if isinstance(d["normalization_factor"], int):
                normalization_factors[idx] = [d["normalization_factor"]] * len(
                    problems[idx]
                )
            elif isinstance(d["normalization_factor"], list):
                normalization_factors[idx] = d["normalization_factor"]
            else:
                raise ValueError(
                    f"Unexpected normalization factor type: {type(d['normalization_factor'])}"
                )
        else:
            normalization_factors[idx] = [-1] * len(problems[idx])

    return problems, normalization_factors


def load_bargaining_table_prompts(
    path: str, player_key: str, variant_tag: str | None = None
):
    with open(path, "r") as f:
        data = json.load(f)

    game = data.get("game", "")
    task = data.get("task", "")
    variants = data.get("variants", None)

    player_key = player_key.lower()
    base_prompt = (
        f"{game}{task}" if not game or game.endswith("\n") else f"{game}\n{task}"
    )
    if "@variant@" in base_prompt:
        if not isinstance(variants, dict) or not variants:
            raise ValueError(
                f"Dataset at {path} includes '@variant@' but has no 'variants' mapping."
            )
        if variant_tag is None:
            raise ValueError(
                f"Dataset at {path} includes '@variant@' but no variant tag was provided."
            )
        if variant_tag not in variants:
            raise ValueError(
                f"Unknown variant '{variant_tag}'. Available variants: {list(variants.keys())}"
            )
        variant_text = variants[variant_tag]
        if variant_text is None:
            variant_text = ""
        if not isinstance(variant_text, str):
            raise ValueError(
                f"Unexpected variant type for '{variant_tag}': {type(variant_text)}"
            )
        if variant_text and not variant_text[0].isspace():
            variant_text = f" {variant_text}"
        base_prompt = base_prompt.replace("@variant@", variant_text)

    problems, normalization_factors = {}, {}
    player_state_keys = {
        key[len("states_") :]: key for key in data.keys() if key.startswith("states_")
    }
    if player_state_keys:
        if "@player@" in base_prompt:
            raise ValueError(
                f"Dataset at {path} uses player-specific states but includes '@player@' in the prompt."
            )
        if player_key not in player_state_keys:
            raise ValueError(
                f"Unknown player '{player_key}'. Available players: {list(player_state_keys.keys())}"
            )
        states = data.get(player_state_keys[player_key], {})
        if not isinstance(states, dict):
            raise ValueError(
                f"Expected '{player_state_keys[player_key]}' to be a dictionary."
            )
        for state_key, state_text in states.items():
            prompt = base_prompt.replace("@state@", state_text)
            problems[state_key] = [prompt]
            normalization_factors[state_key] = [1]
        return problems, normalization_factors

    if "states" in data:
        players = data.get("players", {})
        states = data.get("states", {})
        requires_player = "@player@" in base_prompt

        if not isinstance(players, dict):
            raise ValueError("Expected 'players' to be a dictionary.")
        if not isinstance(states, dict):
            raise ValueError("Expected 'states' to be a dictionary.")
        if requires_player:
            if not players:
                raise ValueError(
                    f"Dataset at {path} includes '@player@' but has no 'players' mapping."
                )
            if player_key not in players:
                raise ValueError(
                    f"Unknown player '{player_key}'. Available players: {list(players.keys())}"
                )
            player_text = players[player_key]
        else:
            player_text = ""

        for state_key, state_text in states.items():
            prompt = base_prompt.replace("@player@", player_text).replace(
                "@state@", state_text
            )
            problems[state_key] = [prompt]
            normalization_factors[state_key] = [1]
        return problems, normalization_factors

    raise ValueError(
        f"Dataset at {path} must include either 'states' or 'states_<player>' keys."
    )


def compute_coordination_index(items: dict, normalization_factor: float = 1.0):
    freq_counter = Counter(items)
    total = sum(freq_counter.values())
    coordination_index = sum([v * (v - 1) for v in dict(freq_counter).values()]) / (
        total * (total - 1)
    )
    normalised_coordination_index = coordination_index * normalization_factor

    return coordination_index, normalised_coordination_index
