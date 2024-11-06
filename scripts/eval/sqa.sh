#!/bin/bash
CKPT=checkpoints/finetune-eagle-x1-llama3.2-1b/checkpoint-2
NAME=EAGLE_SQA

python -m eagle.eval.model_vqa_science \
    --model-path $CKPT \
    --question-file ./dataset/Image_Eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./dataset/Image_Eval/scienceqa/images \
    --answers-file ./dataset/Image_Eval/scienceqa/answers/${NAME}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode llama3

python eagle/eval/eval_science_qa.py \
    --base-dir ./dataset/Image_Eval/scienceqa \
    --result-file ./dataset/Image_Eval/scienceqa/answers/${NAME}.jsonl \
    --output-file ./dataset/Image_Eval/scienceqa/answers/${NAME}_output.jsonl \
    --output-result ./dataset/Image_Eval/scienceqa/answers/${NAME}_result.json
