import argparse
import glob
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import re
from collections import defaultdict
from pathlib import Path


def sample(d: dict, n: int, replacement: bool = True) -> dict:

    if replacement:
        if not d or n >= sum(d.values()):
            return d

    # Create the full population list
    population = []
    for key, count in d.items():
        population.extend([key] * count)

    # Adjust n for no-replacement
    if not replacement and n > len(population):
        n = len(population)

    # Sample from the population
    samples = np.random.choice(population, size=n, replace=replacement)

    # Count occurrences
    result = {}
    for s in samples:
        result[str(s)] = result.get(s, 0) + 1

    return result


def get_family(model_name):
    return (
        model_name.split("/")[1]
        if model_name.startswith("/")
        else model_name.split("/")[0]
    )


def get_color(model_name, models_in_family):
    family = get_family(model_name)
    cmap = family_colormaps[family]
    print(model_name, models_in_family)
    idx = models_in_family.index(model_name)
    n = len(models_in_family)
    return cmap(0.3 + 0.6 * idx / (n - 1 if n > 1 else 1))  # spaced shades


def size_in_billions(model_name: str) -> float:
    """
    Extract the numeric size and unit from model_name and return size in billions.
    Examples matched: -1B, -1.5B, -0.5B, -72B, -14B, -3B, -1500M, etc.
    Falls back to 0.0 if no size token found.
    """
    # Look for a number (integer or decimal) followed by a unit letter (B/M/K) with word boundary.
    m = re.search(r"-(\d+(?:\.\d+)?)([bBkKmM])\b", model_name)
    if not m:
        return 0.0

    value = float(m.group(1))
    unit = m.group(2).upper()

    if unit == "B":
        return value
    if unit == "M":
        return value / 1000.0
    if unit == "K":
        return value / 1_000_000.0

    return value  # fallback (shouldn't reach)


def sort_models_by_size(models, descending=True):
    return sorted(models, key=size_in_billions, reverse=descending)


def get_available_models(model_names, required_files):
    base = Path("./results")
    working_model_names = []
    corrupted_model_names = {}
    required = set(required_files)

    for m in model_names:
        # Ignore entries like ".DS_Store" that slipped into model_names
        if Path(m).name.startswith("."):
            continue

        model_dir = base / m.lstrip("/")  # handle leading slash in m
        # Collect only real, non-hidden files
        files = {
            p.name
            for p in model_dir.glob("*")
            if p.is_file() and not p.name.startswith(".")
        }

        missing_files = required - files
        if missing_files:
            corrupted_model_names[m] = missing_files
        else:
            working_model_names.append(m)

    if corrupted_model_names:
        print("\n--- Corrupted Models (missing files) ---")
        for m, missing in corrupted_model_names.items():
            print(f"Model: {m}")
            print(f"Missing files: {missing}")
            print()
        raise ValueError("Some models are corrupted. See above for details.")

    return working_model_names


def build_required_filename_lookup(required_files, dataset_names, labels):
    lookup = {}
    for file_name in required_files:
        if "_problem-" not in file_name:
            continue
        prefix, suffix = file_name.rsplit("_problem-", 1)
        label = suffix.replace(".jsonl", "")
        dataset_candidate = prefix.split("-")[0]
        key = (dataset_candidate, label)
        if key in lookup and lookup[key] != file_name:
            raise ValueError(
                f"Ambiguous required file mapping for {key}: {lookup[key]} vs {file_name}"
            )
        lookup[key] = file_name

    for dataset in dataset_names:
        for label in labels:
            key = (dataset, label)
            if key not in lookup:
                raise ValueError(
                    f"No required file specified for dataset '{dataset}' and label '{label}'."
                )

    return lookup


# Define families and assign colormaps
family_colormaps = defaultdict(
    lambda: cm.Greys,  # default colormap
    {
        "meta-llama": cm.Blues,
        "google": cm.Reds,
        "microsoft": cm.Greens,
        "Qwen": cm.Purples,
        "deepseek-ai": cm.Oranges,
        "openai": cm.Reds,
        "all-models": cm.viridis,
    },
)


def get_args():
    parser = argparse.ArgumentParser(
        description="Run dataset analysis or model evaluation."
    )

    # Dataset and task options
    parser.add_argument(
        "--dataset-names",
        nargs="+",
        default=["amsterdam", "nottingham"],
        choices=["amsterdam", "nottingham"],
        help="List of dataset names to process.",
    )

    parser.add_argument(
        "--labels",
        nargs="+",
        default=["pick", "guess", "coordinate"],
        choices=["pick", "guess", "coordinate"],
        help="Task labels corresponding to problem types.",
    )

    parser.add_argument(
        "--task-folder",
        default="all-features",
        choices=["vanilla", "saliency", "all-features", "culture"],
        help="Task folder corresponding to the technique used.",
    )

    # Required files
    parser.add_argument(
        "--required-files",
        nargs="+",
        default=[
            "amsterdam-instruct-all-features_problem-coordinate.jsonl",
            "amsterdam-instruct-all-features_problem-guess.jsonl",
            "amsterdam-instruct-all-features_problem-pick.jsonl",
            "nottingham-instruct-all-features_problem-coordinate.jsonl",
            "nottingham-instruct-all-features_problem-guess.jsonl",
            "nottingham-instruct-all-features_problem-pick.jsonl",
        ],
        help="List of required JSONL files for analysis.",
    )

    # Model configurations
    parser.add_argument(
        "--models",
        nargs="+",
        default=["meta-llama"],
        help="Model families (folders under ./results).",
    )

    parser.add_argument(
        "--model-names-selection",
        nargs="+",
        default=[
            "/meta-llama/Meta-Llama-3-70B-Instruct",
            "/meta-llama/Llama-3.1-70B-Instruct",
            "/meta-llama/Llama-3.3-70B-Instruct",
        ],
        help="List of specific model names for plots or evaluation.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    # Argparse options
    args = get_args()
    dataset_names = args.dataset_names
    labels = args.labels
    task_folder = args.task_folder
    required_files = args.required_files
    model_names_selection = args.model_names_selection
    models = args.models
    required_file_lookup = build_required_filename_lookup(
        required_files, dataset_names, labels
    )

    # Other options
    num_samples = 1000
    sample_with_replacement = True
    mode_variations = [("-all-variations", 3)]

    # Matplotlib config
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["font.family"] = "Times New Roman"

    # Create save directories
    SAVEDIR = "./plots"
    sample_folder = (
        "sample-with-replacement"
        if sample_with_replacement
        else "sample-without-replacement"
    )

    assert len(models) == 1, "Please select only one family of models at a time."

    COORDINATION_INDEX_FOLDER = (
        f"/{task_folder}/coordination-index/{sample_folder}/{models[0]}"
    )
    BEST_MODELS_COORDINATION_INDEX_FOLDER = (
        f"/{task_folder}/coordination-index-best/{sample_folder}/{models[0]}"
    )
    BEST_MODELS_COORDINATION_INDEX_SAMPLING_FOLDER = (
        f"/{task_folder}/coordination-index-best-merge/{sample_folder}/{models[0]}"
    )

    for folder in [
        # COORDINATION_INDEX_FOLDER,
        BEST_MODELS_COORDINATION_INDEX_FOLDER,
        BEST_MODELS_COORDINATION_INDEX_SAMPLING_FOLDER,
    ]:
        if not os.path.exists(Path(SAVEDIR + folder)):
            print("Creating path: ", Path(SAVEDIR + folder))
            os.makedirs(Path(SAVEDIR + folder))

    # # 1. Coordination Index -- All Models
    # #Collect the names of the folders inside each "model"
    # model_names = []
    # for m in models:
    #     for model_path in next(os.walk("./results/" + m)):
    #         if "./results/" in model_path:
    #             prefix = model_path
    #             continue
    #         model_names.extend([prefix.split("./results")[1] + "/" + llm for llm in model_path])

    # print("--- Available Models ---")
    # for i,m in enumerate(model_names):
    #     print(f"{i+1}. {m}")

    # model_names = get_available_models(model_names, required_files)

    # # Sort by model-size
    # print(model_names)
    # model_by_size = sort_models_by_size(model_names)

    # family_models = defaultdict(list)
    # for m in model_names:
    #     family_models[get_family(m)].append(m)

    # for suffix, mega_i in mode_variations:
    #     for d_name in dataset_names:
    #         print(f"Dataset: {d_name}")

    #         # Load the human results (we need them for the normalisation factor)
    #         with open(Path("./data/amsterdam_nottingham_schelling") / f"{d_name}.jsonl", "r") as f:
    #             data_humans = json.load(f)
    #         normalization_factors = [d["normalization_factor"] for d in data_humans]

    #         for l in labels:
    #             print(f"Label: {l}")
    #             required_filename = required_file_lookup[(d_name, l)]
    #             # Load LLM results
    #             data_llms = {}
    #             for model_name in model_names:
    #                 print(model_name)
    #                 model_path = Path("./results") / model_name.lstrip("/")
    #                 with open(model_path / required_filename, "r") as f:
    #                     data_llm = json.load(f)

    #                 data_llms[model_name] = []
    #                 for i in range(0, len(data_llm), 3):
    #                     current_responses = {}
    #                     for d in data_llm[i:i+mega_i]:
    #                         # For each task, put together the tasks and then average
    #                         for response in d["responses"].keys():
    #                             match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)

    #                             if match:
    #                                 re_response = match.group(1).strip().lower()
    #                                 if re_response not in current_responses:
    #                                     current_responses[re_response] = 0
    #                                 current_responses[re_response] += d["responses"][response]

    #                     # Normalised coordination index
    #                     current_responses = sample(current_responses, n=num_samples, replacement=sample_with_replacement)
    #                     N = sum(list(current_responses.values()))
    #                     n = normalization_factors[i%3]
    #                     nci = 0.
    #                     if N > 1:
    #                         for _,v in current_responses.items():
    #                             nci += v*(v-1)
    #                         data_llms[model_name].append(nci*n/(N*(N-1)))
    #                     else:
    #                         data_llms[model_name].append(0.)

    #             # Load Human results
    #             with open(f"./data/Bardsley-humans/{d_name}.jsonl", "r") as f:
    #                 results_humans = json.load(f)
    #             current_data_humans = [
    #                 d["normalised_coordination_index"] for d in results_humans if d["task"] == l
    #             ]

    #             # ---- Plot ----
    #             plt.figure(figsize=(12, 5))

    #             tasks = [f"T{d_name[0].upper()}{i+1}" for i in range(14)]
    #             x = np.arange(len(tasks))  # label locations
    #             width = 0.1  # width of each bar

    #             # Plot human data
    #             plt.bar(x - width, current_data_humans, width, label="Humans", color="black")

    #             # Plot LLM data
    #             for idx, model_name in enumerate(model_names):
    #                 current_data_llm = data_llms[model_name]
    #                 family = get_family(model_name)
    #                 color = get_color(model_name, family_models[family])
    #                 plt.bar(x + width*(idx), current_data_llm, width, label=model_name, color=color, edgecolor="black")

    #             plt.xticks(x, tasks)
    #             plt.title(f"{d_name.capitalize()} {l.capitalize()}", fontsize=14)
    #             plt.ylabel("Normalised Coordination Index (NCI)", fontsize=12)
    #             plt.legend(bbox_to_anchor=(1., 1.))
    #             plt.tight_layout()
    #             plt.grid(axis='y', alpha=0.3)
    #             plt.savefig(SAVEDIR + COORDINATION_INDEX_FOLDER + f"/{d_name}-{l}{suffix}.png")
    #             plt.show()

    # # 2. Coordination Index -- Best models
    family_models = defaultdict(list)
    for m in model_names_selection:
        family_models[get_family(m)].append(m)

    for suffix, mega_i in mode_variations:
        for d_name in dataset_names:
            print(f"Dataset: {d_name}")

            # Load the human results (we need them for the normalisation factor)
            with open(Path("./data/amsterdam_nottingham_schelling") / f"{d_name}.jsonl", "r") as f:
                data_humans = json.load(f)
            normalization_factors = [d["normalization_factor"] for d in data_humans]

            for l in labels:
                print(f"Label: {l}")
                required_filename = required_file_lookup[(d_name, l)]
                # Load LLM results
                data_llms = {}
                for model_name in model_names_selection:
                    print(model_name)
                    model_path = Path("./results") / model_name.lstrip("/")
                    with open(model_path / required_filename, "r") as f:
                        data_llm = json.load(f)

                    data_llms[model_name] = []
                    for i in range(0, len(data_llm), 3):
                        current_responses = {}
                        for d in data_llm[i:i+mega_i]:
                            # For each task, put together the tasks and then average
                            for response in d["responses"].keys():
                                match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)

                                if match:
                                    re_response = match.group(1).strip().lower()
                                    if re_response not in current_responses:
                                        current_responses[re_response] = 0
                                    current_responses[re_response] += d["responses"][response]

                        # Normalised coordination index
                        current_responses = sample(current_responses, n=num_samples, replacement=sample_with_replacement)
                        N = sum(list(current_responses.values()))
                        n = normalization_factors[i%3]
                        nci = 0.
                        if N > 1:
                            for _,v in current_responses.items():
                                nci += v*(v-1)
                            data_llms[model_name].append(nci*n/(N*(N-1)))
                        else:
                            data_llms[model_name].append(0.)

                # Load Human results
                with open(f"./data/Bardsley-humans/{d_name}.jsonl", "r") as f:
                    results_humans = json.load(f)
                current_data_humans = [
                    d["normalised_coordination_index"] for d in results_humans if d["task"] == l
                ]

                # ---- Plot ----
                plt.figure(figsize=(12, 5))

                tasks = [f"T{d_name[0].upper()}{i+1}" for i in range(14)]
                x = np.arange(len(tasks))  # label locations
                width = 0.1  # width of each bar

                # Plot human data
                plt.bar(x - width, current_data_humans, width, label="Humans", color="black")

                # Plot LLM data
                for idx, model_name in enumerate(model_names_selection):
                    current_data_llm = data_llms[model_name]
                    family = get_family(model_name)
                    color = get_color(model_name, family_models[family])
                    plt.bar(x + width*(idx), current_data_llm, width, label=model_name, color=color, edgecolor="black")

                plt.xticks(x, tasks)
                plt.title(f"{d_name.capitalize()} {l.capitalize()}", fontsize=14)
                plt.ylabel("Normalised Coordination Index (NCI)", fontsize=12)
                # plt.legend(bbox_to_anchor=(1., 1.))
                plt.tight_layout()
                plt.grid(axis='y', alpha=0.3)
                plt.savefig(SAVEDIR + BEST_MODELS_COORDINATION_INDEX_FOLDER + f"/{d_name}-{l}{suffix}.png")
                plt.show()

    # 3. Coordination Index -- Merge best models
    family_models = defaultdict(list)
    for m in model_names_selection:
        family_models[get_family(m)].append(m)

    for suffix, mega_i in mode_variations:
        for d_name in dataset_names:
            print(f"Dataset: {d_name}")
            # Load the human results (we need them for the normalisation factor)
            with open(Path("./data/amsterdam_nottingham_schelling") / f"{d_name}.jsonl", "r") as f:
                data_humans = json.load(f)
            normalization_factors = [d["normalization_factor"] for d in data_humans]

            for l in labels:
                data_llms = []
                print(f"Label: {l}")
                required_filename = required_file_lookup[(d_name, l)]

                # Load LLM results
                for i in range(0, len(data_humans) * 3, 3):  # TA
                    current_responses = {}
                    for model_name in model_names_selection:
                        model_path = Path("./results") / model_name.lstrip("/")
                        with open(model_path / required_filename, "r") as f:
                            data_llm = json.load(f)

                        for d in data_llm[i : i + mega_i]:
                            # For each task, put together the tasks and then average
                            for response in d["responses"].keys():
                                match = re.search(
                                    r"<answer>(.*?)</answer>", response, re.DOTALL
                                )

                                if match:
                                    re_response = match.group(1).strip().lower()
                                    if re_response not in current_responses:
                                        current_responses[re_response] = 0
                                    current_responses[re_response] += d["responses"][
                                        response
                                    ]

                    # Normalised coordination index
                    current_responses = sample(
                        current_responses,
                        n=num_samples,
                        replacement=sample_with_replacement,
                    )
                    N = sum(list(current_responses.values()))
                    n = normalization_factors[i % 3]
                    nci = 0.0
                    if N > 1:
                        for _, v in current_responses.items():
                            nci += v * (v - 1)
                        data_llms.append(nci * n / (N * (N - 1)))
                    else:
                        data_llms.append(0.0)

                # Load Human results
                with open(f"./data/Bardsley-humans/{d_name}.jsonl", "r") as f:
                    results_humans = json.load(f)
                current_data_humans = [
                    dd["normalised_coordination_index"]
                    for dd in results_humans
                    if dd["task"] == l
                ]

                # ---- Plot ----
                plt.figure(figsize=(12, 5))

                tasks = [f"T{d_name[0].upper()}{i+1}" for i in range(14)]
                x = np.arange(len(tasks))  # label locations
                width = 0.3  # width of each bar

                human_avg = (
                    float(np.mean(current_data_humans)) if current_data_humans else 0.0
                )
                llm_avg = float(np.mean(data_llms)) if data_llms else 0.0
                human_color = "black"
                llm_family = models[0] if models else "LLM"
                family_cmap = family_colormaps[llm_family]
                llm_color = family_cmap(0.6)
                llm_label = llm_family

                # Plot human data (shifted left)
                human_bars = plt.bar(
                    x - width / 2,
                    current_data_humans,
                    width,
                    label="Humans",
                    color=human_color,
                )

                # Plot meta-llama data (shifted right)
                llm_bars = plt.bar(
                    x + width / 2,
                    data_llms,
                    width,
                    label=llm_label,
                    color=llm_color,
                    edgecolor="black",
                )

                # Add values on top of human bars
                for bar in human_bars:
                    height = bar.get_height()
                    plt.text(
                        bar.get_x() + bar.get_width() / 2,
                        height + 0.02,
                        f"{height:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

                # Add values on top of LLM bars
                for bar in llm_bars:
                    height = bar.get_height()
                    plt.text(
                        bar.get_x() + bar.get_width() / 2,
                        height + 0.02,
                        f"{height:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

                ax = plt.gca()
                plt.axhline(human_avg, color=human_color, linestyle="--", linewidth=1.2)
                plt.axhline(llm_avg, color=llm_color, linestyle="--", linewidth=1.2)
                x_text = ax.get_xlim()[1] - 0.2
                text_kwargs = {"ha": "right", "va": "bottom", "fontsize": 8}
                plt.text(
                    x_text,
                    human_avg,
                    f"Human avg: {human_avg:.2f}",
                    color=human_color,
                    **text_kwargs,
                )
                plt.text(
                    x_text,
                    llm_avg,
                    f"{llm_label} avg: {llm_avg:.2f}",
                    color=llm_color,
                    **text_kwargs,
                )

                plt.xticks(x, tasks)
                plt.title(f"{d_name.capitalize()} {l.capitalize()}", fontsize=14)
                plt.ylabel("Normalised Coordination Index (NCI)", fontsize=12)
                # plt.legend(bbox_to_anchor=(1.0, 1.0))
                plt.tight_layout()
                plt.grid(axis="y", alpha=0.3)
                plt.savefig(
                    SAVEDIR
                    + BEST_MODELS_COORDINATION_INDEX_SAMPLING_FOLDER
                    + f"/{d_name}-{l}{suffix}.png"
                )
                plt.show()
