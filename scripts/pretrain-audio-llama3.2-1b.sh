#!/bin/bash
NAME=audio_try
export WANDB_DISABLED="true"

CUDA_VISIBLE_DEVICES='6,7' python -m torch.distributed.run \
    --nproc_per_node 2 --master_port 25031 \
    train_audio.py \
    --model_name_or_path ./model/LLM/Llama-3.2-1B-Instruct \
    --version plain \
    --data_path ./dataset/Audio/AudioSetCaps/example.csv \
    --audio_folder ./dataset/Audio/AudioSetCaps/example \
    --audio_tower "./model/LanguageBind_Audio_FT" \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_audio_projection True \
    --mm_audio_select_layer -2 \
    --bf16 True \
    --fp16 False \
    --output_dir ./checkpoints/$NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
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
    --run_name ${NAME}
