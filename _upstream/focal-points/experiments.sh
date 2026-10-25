#!/bin/bash

# Default arguments
default_models=("meta-llama/Llama-3.3-70B-Instruct")
models=()
datasets=()
num_experiments=30
quantization="None"
plot_graphs="true"
reasoning=""
max_new_tokens=""
bargaining_players=("blue" "yellow")

# Parse command-line arguments
while getopts "m:d:n:q:p:r:x:b:" opt; do
  case $opt in
    m)
      read -r -a parsed_models <<< "$OPTARG"
      models+=("${parsed_models[@]}")
      ;;
    d)
      read -r -a parsed_datasets <<< "$OPTARG"
      datasets+=("${parsed_datasets[@]}")
      ;;
    n)
      num_experiments=$OPTARG
      ;;
    q)
      quantization=$OPTARG
      ;;
    p)
      plot_graphs=$OPTARG
      ;;
    r)
      reasoning=$OPTARG
      ;;
    x)
      max_new_tokens=$OPTARG
      ;;
    b)
      read -r -a parsed_players <<< "$OPTARG"
      bargaining_players=("${parsed_players[@]}")
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

available_models=(
  "meta-llama/Llama-3.3-70B-Instruct"
  "meta-llama/Llama-3.1-70B-Instruct"
  "meta-llama/Meta-Llama-3-70B-Instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
  "meta-llama/Meta-Llama-3-8B-Instruct"
  "meta-llama/Llama-3.2-3B-Instruct"
  "meta-llama/Llama-3.2-1B-Instruct"
  "Qwen/Qwen2-72B-Instruct"
  "Qwen/Qwen2.5-72B-Instruct"
  "Qwen/Qwen2.5-32B-Instruct"
  "Qwen/Qwen2.5-14B-Instruct-1M"
  "Qwen/Qwen2.5-14B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct-1M"
  "Qwen/Qwen2.5-7B-Instruct"
  "Qwen/Qwen2-7B-Instruct"
  "Qwen/Qwen2.5-3B-Instruct"
  "Qwen/Qwen2.5-1.5B-Instruct"
  "Qwen/Qwen2-1.5B-Instruct"
  "Qwen/Qwen2.5-0.5B-Instruct"
  "Qwen/Qwen2-0.5B-Instruct"
  "google/gemma-3-1b-it"
  "google/gemma-3-4b-it"
  "google/gemma-3-12b-it"
  "google/gemma-3-27b-it"
  "microsoft/Phi-4-mini-instruct"
  "openai/gpt-oss-120b"
  "openai/gpt-oss-20b"
  "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
)

all_datasets=(
  # TASK -- Amsterdam
  "amsterdam"
  "amsterdam-instruct-all-features"
  "amsterdam-instruct-saliency"
  # TASK -- Amsterdam_numeric
  "amsterdam_numeric"
  "amsterdam_numeric-instruct-all-features"
  "amsterdam_numeric-instruct-saliency"
  # TASK -- Asymmetric_payoff
  "asymmetric_payoff"
  "asymmetric_payoff-instruct-all-features"
  "asymmetric_payoff-instruct-saliency"
  # TASK -- Nottingham
  "nottingham"
  "nottingham-instruct-all-features"
  "nottingham-instruct-saliency"
  # TASK -- Nottingham_numeric
  "nottingham_numeric"
  "nottingham_numeric-instruct-all-features"
  "nottingham_numeric-instruct-saliency"
  # TASK -- Amsterdam culture
  "amsterdam-instruct-culture"
  # TASK -- Nottingham culture
  "nottingham-instruct-culture"
  # schelling
  "schelling"
  "schelling-instruct-all-features"
  "schelling-instruct-saliency"
  # bargaining table
  "bargaining_table_realdata"
  )

# Apply defaults if none provided
if ((${#models[@]} == 0)); then
  models=("${default_models[@]}")
fi

if ((${#datasets[@]} == 0)); then
  datasets=("${all_datasets[@]}")
fi

# Handle "all" sentinel
if ((${#models[@]} == 1)) && [[ "${models[0]}" == "all" ]]; then
  models=("${available_models[@]}")
fi

if ((${#datasets[@]} == 1)) && [[ "${datasets[0]}" == "all" ]]; then
  datasets=("${all_datasets[@]}")
fi

standard_problemtags=("problem-pick" "problem-guess" "problem-coordinate")
schelling_problemtags=("problem")
bargaining_problemtags=("greedy" "cooperative" "all-features" "saliency" "vanilla")
standard_datasets=()
schelling_datasets=()
bargaining_datasets=()

for data in "${datasets[@]}"; do
  case "$data" in
    "schelling"|"schelling-instruct-all-features"|"schelling-instruct-saliency")
      schelling_datasets+=("$data")
      ;;
    "bargaining_table"|"bargaining_table_realdata")
      bargaining_datasets+=("$data")
      ;;
    *)
      standard_datasets+=("$data")
      ;;
  esac
done

reasoning_display="${reasoning:-Default}"
echo "Running experiments with model(s): ${models[*]}, datasets: ${datasets[*]}, quantization: $quantization, number of experiments: $num_experiments, reasoning: $reasoning_display, bargaining players: ${bargaining_players[*]}"

extra_args=()
if [[ -n "$max_new_tokens" && "$max_new_tokens" != "None" && "$max_new_tokens" != "none" ]]; then
  extra_args=(--max-new-tokens "$max_new_tokens")
fi

reasoning_args=()
if [[ -n "$reasoning" ]]; then
  reasoning_args=(--reasoning "$reasoning")
fi

for model in "${models[@]}"; do
  echo "Running experiments for model: $model"

  if ((${#standard_datasets[@]})); then
    echo "Running main.py for standard datasets: ${standard_datasets[*]}"
    python main.py --model "$model" --dataset "${standard_datasets[@]}" --problem-tag "${standard_problemtags[@]}" --return-sequences "$num_experiments" --quantization "$quantization" --plot-graphs "$plot_graphs" "${reasoning_args[@]}" "${extra_args[@]}"
  fi

  if ((${#schelling_datasets[@]})); then
    echo "Running main.py for schelling datasets: ${schelling_datasets[*]}"
    python main.py --model "$model" --dataset "${schelling_datasets[@]}" --problem-tag "${schelling_problemtags[@]}" --return-sequences "$num_experiments" --quantization "$quantization" --plot-graphs "$plot_graphs" "${reasoning_args[@]}" "${extra_args[@]}"
  fi

  if ((${#bargaining_datasets[@]})); then
    for bargaining_player in "${bargaining_players[@]}"; do
      echo "Running main.py for bargaining table datasets: ${bargaining_datasets[*]} (player: $bargaining_player)"
      python main.py --model "$model" --dataset "${bargaining_datasets[@]}" --problem-tag "${bargaining_problemtags[@]}" --return-sequences "$num_experiments" --quantization "$quantization" --plot-graphs "$plot_graphs" "${reasoning_args[@]}" --bargaining-player "$bargaining_player" "${extra_args[@]}"
    done
  fi
done
