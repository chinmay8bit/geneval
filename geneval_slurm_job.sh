#!/bin/bash

#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=resgpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cp524

set -euo pipefail

export HF_HOME="/vol/bitbucket/cp524/hf_cache"

# for offline loading only
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

# Activate virtual environment
# export PATH=/vol/bitbucket/cp524/dev/papers_with_code/geneval/venv/bin:$PATH
# source /vol/bitbucket/cp524/dev/papers_with_code/geneval/venv/bin/activate

# Set up CUDA
# source /vol/cuda/12.5.0/setup.sh

# Navigate to script directory
cd /vol/bitbucket/cp524/dev/papers_with_code/geneval
export PYTHONPATH=${PYTHONPATH:-}:$(pwd)

export PYTHONUNBUFFERED=1

# Base directories
INPUT_BASE="/vol/bitbucket/cp524/dev/papers_with_code/Fk-Diffusion-Steering/text_to_image/geneval_metadata_outputs"
# OUTPUT_BASE="/vol/bitbucket/cp524/dev/papers_with_code/Fk-Diffusion-Steering/text_to_image/geneval_metadata_results"
OUTPUT_BASE="/vol/bitbucket/cp524/dev/papers_with_code/Fk-Diffusion-Steering/text_to_image/geneval_metadata_results_best"
# INPUT_BASE="/vol/bitbucket/cp524/dev/SMC_Meissonic_2/geneval_metadata_outputs"
# OUTPUT_BASE="/vol/bitbucket/cp524/dev/SMC_Meissonic_2/geneval_metadata_results"
# OUTPUT_BASE="/vol/bitbucket/cp524/dev/SMC_Meissonic_2/geneval_metadata_results_best"

OBJECT_DETECTOR_FOLDER="object_detector"

# Make sure the output base exists
mkdir -p "$OUTPUT_BASE"

# Loop over every subdirectory in INPUT_BASE
for IMAGE_FOLDER in "$INPUT_BASE"/*/; do
  # Strip trailing slash and extract just the folder name
  FOLDER_NAME="$(basename "${IMAGE_FOLDER%/}")"
  RESULTS_FOLDER="$OUTPUT_BASE/$FOLDER_NAME"
  RESULTS_FILE="$RESULTS_FOLDER/results.jsonl"
  SUMMARY_FILE="$RESULTS_FOLDER/summary.json"

  echo "=== Processing folder: $FOLDER_NAME ==="

  # Ensure the results folder exists
  mkdir -p "$RESULTS_FOLDER"

  if [[ -f "$RESULTS_FILE" ]]; then
    echo "→ results.jsonl already exists; skipping evaluation step."
  else
    echo "→ Running evaluation step..."
    python evaluation/evaluate_images.py \
      "$IMAGE_FOLDER" \
      --outfile "$RESULTS_FILE" \
      --model-path "$OBJECT_DETECTOR_FOLDER" \
      --samples-dir "best_of_n_samples"
  fi

  echo "→ Generating summary..."
  python evaluation/summary_scores.py "$RESULTS_FILE" \
    --outfile "$SUMMARY_FILE"

  echo "→ Done with $FOLDER_NAME"
  echo
done

echo "All folders processed."
