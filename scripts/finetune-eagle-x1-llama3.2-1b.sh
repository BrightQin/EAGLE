#!/bin/bash
NAME=finetune-eagle-x1-llama3.2-1b

# export WANDB_DISABLED="true"
export WANDB_PROJECT="eagle"
export WANDB_RUN_ID=${NAME}
export WANDB_RESUME="allow"

CUDA_VISIBLE_DEVICES='6' python -m torch.distributed.run \
    --nproc_per_node 1 --master_port 25031 \
    train_mem.py \
    --model_name_or_path ./model/LLM/Llama-3.2-1B-Instruct \
    --version llama3 \
    --data_path ./dataset/Eagle-1.8M/eagle-1-sft-1_8M.json \
    --image_folder ./dataset/Eagle-1.8M \
    --vision_tower ./model/Vision_Encoder/openai/TinyCLIP-ViT-40M-32-Text-19M-LAION400M \
    --mm_projector_type mlp2x_gelu \
    --pretrain_mm_mlp_adapter ./checkpoints/pretrain-eagle-x1-llama3.2-1b/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 False \
    --fp16 True \
    --output_dir ./checkpoints/$NAME \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 512 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --run_name ${NAME}  
