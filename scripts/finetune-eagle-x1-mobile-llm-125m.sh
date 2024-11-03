#!/bin/bash
NAME=$1

# export WANDB_DISABLED="true"
export WANDB_PROJECT="eagle"
export WANDB_RUN_ID=${NAME}
export WANDB_RESUME="allow"

CUDA_VISIBLE_DEVICES='5,6' python -m torch.distributed.run \
    --nproc_per_node 2 --master_port 25031 \
    train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ./model/LLM/MobileLLM-125M \
    --version v0.5 \
    --data_path ./dataset/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder ./dataset/LLaVA-Pretrain \
    --vision_tower ./model/Vision_Encoder/openai/clip-vit-base-patch32 \
    --mm_projector_type mlp2x_gelu \
    --pretrain_mm_mlp_adapter ./checkpoints/pretrain-eagle-x1-mobile-llm-125m/checkpoint-20/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 False \
    --fp16 True \
    --output_dir ./checkpoints/$NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --run_name ${NAME}  
