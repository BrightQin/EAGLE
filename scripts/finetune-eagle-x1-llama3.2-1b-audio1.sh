#!/bin/bash
NAME=audio_finetune

export WANDB_DISABLED="true"
export WANDB_PROJECT="eagle"
export WANDB_RUN_ID=${NAME}
export WANDB_RESUME="allow"

# echo "MASTER_ADDR=$MASTER_ADDR"
# n_node=$SLURM_JOB_NUM_NODES
# echo "number of nodes:" $n_node
# echo "node rank:" $SLURM_PROCID

CUDA_VISIBLE_DEVICES='4' python -m torch.distributed.run \
    --nproc_per_node 1 --master_port 25031 \
    train_audio1.py \
    --model_name_or_path ./model/LLM/Llama-3.2-1B-Instruct \
    --version llama3 \
    --data_path ./dataset/Audio/AudioSetCaps/example.json \
    --image_folder ./dataset/Audio/AudioSetCaps/example \
    --vision_tower ./model/LanguageBind_Audio_FT \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 False \
    --fp16 True \
    --output_dir ./checkpoints/$NAME \
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
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --run_name ${NAME} \
    --max_steps 2 \
    --save_safetensors False
    # --num_train_epochs 2 \
