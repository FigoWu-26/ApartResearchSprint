#!/bin/bash
set -u
set -o pipefail
# note: we do NOT set -e so we can handle python failures manually

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

# Configure which model families to plot (default: meta-llama). You can also pass families as CLI args.
# MODELS=("meta-llama" "Qwen" "all-models")
MODELS=("meta-llama" "Qwen" "openai-120b" "openai-20b" "all-models")
if [ $# -gt 0 ]; then
    MODELS=("$@")
fi

DATASET_NAMES=("amsterdam" "nottingham")
LABELS=("pick" "guess" "coordinate")
TASK_FOLDER=("vanilla" "saliency" "all-features" "culture")

META_LLAMA_MODELS=(
  "/meta-llama/Meta-Llama-3-70B-Instruct"
  "/meta-llama/Llama-3.1-70B-Instruct"
  "/meta-llama/Llama-3.3-70B-Instruct"
)

QWEN_MODELS=(
  "/Qwen/Qwen2-72B-Instruct"
  "/Qwen/Qwen2.5-72B-Instruct"
)

OPENAI_120B_MODELS=(
  "/openai/gpt-oss-120b_low"
  "/openai/gpt-oss-120b_medium"
  "/openai/gpt-oss-120b_high"
)

OPENAI_20B_MODELS=(
  "/openai/gpt-oss-20b_low"
  "/openai/gpt-oss-20b_medium"
  "/openai/gpt-oss-20b_high"
)

META_LLAMA_MODELS=(
  "/meta-llama/Meta-Llama-3-70B-Instruct"
  "/meta-llama/Llama-3.1-70B-Instruct"
  "/meta-llama/Llama-3.3-70B-Instruct"
)

QWEN_MODELS=(
  "/Qwen/Qwen2-72B-Instruct"
  "/Qwen/Qwen2.5-72B-Instruct"
)

ALL_MODELS=(
  "/meta-llama/Meta-Llama-3-70B-Instruct"
  "/meta-llama/Llama-3.1-70B-Instruct"
  "/meta-llama/Llama-3.3-70B-Instruct"
  "/Qwen/Qwen2-72B-Instruct"
  "/Qwen/Qwen2.5-72B-Instruct"
  "/openai/gpt-oss-120b_low"
  "/openai/gpt-oss-120b_medium"
  "/openai/gpt-oss-120b_high"
)

REQUIRED_FILES_VANILLA=(
    "amsterdam_problem-coordinate.jsonl"
    "amsterdam_problem-guess.jsonl"
    "amsterdam_problem-pick.jsonl"
    "nottingham_problem-coordinate.jsonl"
    "nottingham_problem-guess.jsonl"
    "nottingham_problem-pick.jsonl"
)

REQUIRED_FILES_INSTRUCT=(
    "amsterdam-instruct-saliency_problem-coordinate.jsonl"
    "amsterdam-instruct-saliency_problem-guess.jsonl"
    "amsterdam-instruct-saliency_problem-pick.jsonl"
    "nottingham-instruct-saliency_problem-coordinate.jsonl"
    "nottingham-instruct-saliency_problem-guess.jsonl"
    "nottingham-instruct-saliency_problem-pick.jsonl"
)

REQUIRED_FILES_INSTRUCT_ALL_FEATURES=(
    "amsterdam-instruct-all-features_problem-coordinate.jsonl"
    "amsterdam-instruct-all-features_problem-guess.jsonl"
    "amsterdam-instruct-all-features_problem-pick.jsonl"
    "nottingham-instruct-all-features_problem-coordinate.jsonl"
    "nottingham-instruct-all-features_problem-guess.jsonl"
    "nottingham-instruct-all-features_problem-pick.jsonl"
)

REQUIRED_FILES_INSTRUCT_CULTURE=(
    "amsterdam-instruct-culture_problem-coordinate.jsonl"
    "amsterdam-instruct-culture_problem-guess.jsonl"
    "amsterdam-instruct-culture_problem-pick.jsonl"
    "nottingham-instruct-culture_problem-coordinate.jsonl"
    "nottingham-instruct-culture_problem-guess.jsonl"
    "nottingham-instruct-culture_problem-pick.jsonl"
)

PY_SCRIPT="$ROOT_DIR/src/plotting/plot_results.py"
mkdir -p logs

for MODEL in "${MODELS[@]}"; do
    echo "=== Model family: $MODEL ==="
    MODEL_LOWER=$(echo "$MODEL" | tr 'A-Z' 'a-z')

    case "$MODEL_LOWER" in
        "meta-llama")
            SELECTED_MODELS=("${META_LLAMA_MODELS[@]}")
            ;;
        "qwen")
            SELECTED_MODELS=("${QWEN_MODELS[@]}")
            ;;
        "openai-120b")
            SELECTED_MODELS=("${OPENAI_120B_MODELS[@]}")
            ;;
        "openai-20b")
            SELECTED_MODELS=("${OPENAI_20B_MODELS[@]}")
            ;;
        "all-models")
            SELECTED_MODELS=("${ALL_MODELS[@]}")
            ;;
        *)
            echo "Warning: Unknown model family '$MODEL' — skipping."
            continue
            ;;
    esac

    for TASK in "${TASK_FOLDER[@]}"; do
        echo "--- Task folder: $TASK ---"
        case "$TASK" in
            "vanilla")
                REQUIRED_FILES=("${REQUIRED_FILES_VANILLA[@]}")
                ;;
            "saliency")
                REQUIRED_FILES=("${REQUIRED_FILES_INSTRUCT[@]}")
                ;;
            "all-features")
                REQUIRED_FILES=("${REQUIRED_FILES_INSTRUCT_ALL_FEATURES[@]}")
                ;;
            "culture")
                REQUIRED_FILES=("${REQUIRED_FILES_INSTRUCT_CULTURE[@]}")
                ;;
            *)
                echo "Unknown task folder: $TASK — skipping."
                continue
                ;;
        esac

        for DATASET in "${DATASET_NAMES[@]}"; do
            LOG_FILE="logs/${MODEL}_${TASK}_${DATASET}.log"
            echo "Running: $MODEL / $TASK / $DATASET (log -> $LOG_FILE)"
            echo "Prompting files: ${REQUIRED_FILES[@]}"

            # Run Python and explicitly handle failures so loop continues.
            if ! python "$PY_SCRIPT" \
                --dataset-names "$DATASET" \
                --labels "${LABELS[@]}" \
                --task-folder "$TASK" \
                --models "$MODEL" \
                --model-names-selection "${SELECTED_MODELS[@]}" \
                --required-files "${REQUIRED_FILES[@]}" \
                >"$LOG_FILE" 2>&1; then

                echo "⚠️  Python failed for ${MODEL}_${TASK}_${DATASET}"
                echo "---- Last 20 lines of the log ----"
                tail -n 20 "$LOG_FILE" | sed 's/^/    /'
                echo "----------------------------------"
                # continue to next dataset/task/model
                continue
            fi

            echo "✅  Finished ${MODEL}_${TASK}_${DATASET}"
            echo ""
        done
    done
done

echo "✅ All loop iterations attempted (check logs/ for details)."
