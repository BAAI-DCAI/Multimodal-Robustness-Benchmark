#!/bin/bash

MODEL_TYPE=phi-2
TARGET_DIR=-$MODEL_TYPE
ANSWERS_FILE=phi-2-L2M-llava665k-5-llava695k_mmr-5

python -m model_vqa_mmr \
    --model-path /share/project/$ANSWERS_FILE \
    --model-type $MODEL_TYPE \
    --image-folder /share/project/mmr/mmr_v3 \
    --question-file /share/project/mmr/mmr_v3/question_mmr_v5.json \
    --answers-file ./eval/mmr_v3/answers/result/$ANSWERS_FILE.jsonl \
    --temperature 0 \
    --conv-mode bunny

python calculation_mmr.py --result_file answers/result/$ANSWERS_FILE.jsonl | tee 2>&1 results/result/$ANSWERS_FILE.txt