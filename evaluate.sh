#!/bin/bash

export HF_HOME="/vol/bitbucket/cp524/hf_cache"

FOLDER_NAME="20250710-183002"
IMAGE_FOLDER="/vol/bitbucket/cp524/dev/papers_with_code/Fk-Diffusion-Steering/text_to_image/geneval_metadata_outputs/$FOLDER_NAME"
RESULTS_FOLDER="/vol/bitbucket/cp524/dev/papers_with_code/Fk-Diffusion-Steering/text_to_image/geneval_metadata_results/$FOLDER_NAME"
OBJECT_DETECTOR_FOLDER="object_detector"

python evaluation/evaluate_images.py \
    "$IMAGE_FOLDER" \
    --outfile "$RESULTS_FOLDER/results.jsonl" \
    --model-path "$OBJECT_DETECTOR_FOLDER"

python evaluation/summary_scores.py "$RESULTS_FOLDER/results.jsonl"

