#!/bin/bash
NAME=$1

# export WANDB_DISABLED="true"
export WANDB_PROJECT="eagle"
export WANDB_RUN_ID=${NAME}
export WANDB_RESUME="allow"

echo "MASTER_ADDR=$MASTER_ADDR"
n_node=$SLURM_JOB_NUM_NODES
echo "number of nodes:" $n_node
echo "node rank:" $SLURM_PROCID

CUDA_VISIBLE_DEVICES='0' python -m torch.distributed.run \
    --nproc_per_node 1 --master_port 25031 \
    train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ./model/LLM/Meta-Llama-3-8B-Instruct \
    --version llama3 \
    --data_path ./dataset/Eagle-1.8M/eagle-sft-v1-1_8m.json \
    --image_folder ./dataset/Eagle-1.8M/images \
    --vision_tower "clip-448;convnext-1024;sam-1024;det-1024;pix2struct-1024" \
    --mm_projector_type mlp2x_gelu \
    --pretrain_mm_mlp_adapter $PATH_TO_PRETRAINED_PROJECTOR/mm_projector.bin \
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
    --save_steps 500 \
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
    --report_to wandb \
    --run_name ${NAME}  
