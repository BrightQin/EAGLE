#!/bin/bash
NAME=image_try

export WANDB_DISABLED="true"

CUDA_VISIBLE_DEVICES='6,7' python -m torch.distributed.run \
    --nproc_per_node 2 --master_port 25031 \
    train_mem.py \
    --model_name_or_path ./model/LLM/Llama-3.2-1B-Instruct \
    --version plain \
    --data_path ./dataset/LLaVA-Pretrain/blip_laion_cc_sbu_10.json \
    --image_folder ./dataset/LLaVA-Pretrain/images \
    --vision_tower ./model/Vision_Encoder/openai/TinyCLIP-ViT-40M-32-Text-19M-LAION400M \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 False \
    --fp16 True \
    --output_dir ./checkpoints/$NAME \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
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

# python -m torch.distributed.run \
#     --nproc_per_node 8 --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
#     --master_addr $MASTER_ADDR --master_port 25031 \
#     train_mem.py \
#     --deepspeed ./scripts/zero2.json \
#     --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
#     --version plain \
#     --data_path $PATH_TO_PRETRAINING_DATA/blip_laion_cc_sbu_558k.json \
#     --image_folder $PATH_TO_PRETRAINING_DATA/images \
#     --vision_tower "clip-448;convnext-1024;sam-1024;det-1024;pix2struct-1024" \
#     --mm_projector_type mlp2x_gelu \
#     --tune_mm_mlp_adapter True \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --bf16 True \
#     --output_dir ./checkpoints/$NAME \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 24000 \
#     --save_total_limit 1 \
#     --learning_rate 1e-3 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to wandb \
#     --run_name ${NAME}