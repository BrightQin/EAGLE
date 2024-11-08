#!/bin/bash
CKPT=checkpoints/video_finetune
NAME=EAGLE_ACTIVITYNETQA

CUDA_VISIBLE_DEVICES='5,6' python -m eval.eval_activitynet \
    --model-path $CKPT \
    --question-file ./dataset/Video/test/Activitynet_Zero_Shot_QA/test_q.json \
    --image-folder ./dataset/Video/test/Activitynet_Zero_Shot_QA/Test_Videos \
    --answers-file ./output/eval/Activitynet_Zero_Shot_QA/${NAME}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode llama3

# python eagle/eval/eval_science_qa.py \
#     --base-dir ./dataset/Image_Eval/scienceqa \
#     --result-file ./dataset/Image_Eval/scienceqa/answers/${NAME}.jsonl \
#     --output-file ./dataset/Image_Eval/scienceqa/answers/${NAME}_output.jsonl \
#     --output-result ./dataset/Image_Eval/scienceqa/answers/${NAME}_result.json
