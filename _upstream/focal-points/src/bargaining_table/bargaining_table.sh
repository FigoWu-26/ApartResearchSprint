#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY_SCRIPT="$ROOT_DIR/src/bargaining_table/bargaining_table.py"
cd "$ROOT_DIR"

# Traditional strategy experiments
STRATEGIES=(
"humans" 
"p1_greedy"
"p2_greedy"
"both_greedy"  # may break the plots limits!
"p1_cooperative"
"p2_cooperative"
"p1_SVO"
"p2_SVO"
"p1_adaptive"
"p2_adaptive"
)

NUM_SAMPLES=100
SAMPLE_WITH_REPLACEMENT=False
SEED=42
for STRATEGY in "${STRATEGIES[@]}"; do
    echo "[bash] Running bargaining table analysis for strategy: $STRATEGY, num_samples: $NUM_SAMPLES, sample_with_replacement: $SAMPLE_WITH_REPLACEMENT, seed: $SEED"
    python3 "$PY_SCRIPT" --strategy "$STRATEGY" --num-samples "$NUM_SAMPLES" --sample-with-replacement "$SAMPLE_WITH_REPLACEMENT" --seed "$SEED"
done


# LLM experiments
REASONING=(
  "low"
  "medium"
  "high"
)

MODELS=(
  "gpt-oss-20b"
  "gpt-oss-120b"
)

COLORS=(
  "blue"
  "yellow"
)

FILES=(
  "bargaining_table_realdata_responses_vanilla"
  "bargaining_table_realdata_responses_saliency"
  "bargaining_table_realdata_responses_greedy"
  "bargaining_table_realdata_responses_cooperative"
  "bargaining_table_realdata_responses_all-features"
)

declare -A COLOR_TO_STRATEGY=(
  [blue]="p1_llm"
  [yellow]="p2_llm"
)

for color in "${COLORS[@]}"; do
  strategy="${COLOR_TO_STRATEGY[$color]}"

  echo "COLOR=$color STRATEGY=$strategy"
done

STRATEGY="p2_llm"
NUM_SAMPLES=100
SAMPLE_WITH_REPLACEMENT=False
SEED=42

# Create folders for results
mkdir -p "./data/bargaining_table_llms/${color}/${model}-${reasoning}"
mkdir -p "./data/bargaining_table_llms/${color}/${model}-${reasoning}"

for color in "${COLORS[@]}"; do
    STRATEGY="${COLOR_TO_STRATEGY[$color]}"
    for model in "${MODELS[@]}"; do
        for reasoning in "${REASONING[@]}"; do
            for file in "${FILES[@]}"; do
            PATH_FILE="./data/bargaining_table_llms/${color}/${model}-${reasoning}/${file}_${color}.jsonl"
            python3 "$PY_SCRIPT" --strategy "$STRATEGY" --num-samples "$NUM_SAMPLES" --sample-with-replacement "$SAMPLE_WITH_REPLACEMENT" --seed "$SEED" --file-player-as-llm "$PATH_FILE"
            done
        done
    done
done
