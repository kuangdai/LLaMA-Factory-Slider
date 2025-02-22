#!/bin/bash

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Disable version check
export DISABLE_VERSION_CHECK=1

# Define arguments using an array for readability
args=(

    # Command
    torchrun --nproc_per_node=8 --master_port 29505 ./src/train.py

    # Model Settings
    --run_name "full_slider_on"
    --model_name_or_path "/home/shared/base_models/qwen/Qwen2.5-32B-Instruct"
    --finetuning_type full
    --slider_on True
    --slider_n_variables 3
    --slider_n_hidden 256
    --slider_n_heads_sharing_slider 8
    --slider_dropout 0.1

    # Data Settings
    --dataset "test_slider_on"
    --val_size 0.02
    --cutoff_len 4096

    # Output Settings
    --output_dir "test_output/full_slider_on"
    --report_to wandb
    --save_steps 250
    --save_total_limit 2

    # Training Hyperparameters
    --num_train_epochs 5
    --per_device_train_batch_size 4
    --per_device_eval_batch_size 4
    --gradient_accumulation_steps 2
    --learning_rate 1e-4
    --lr_scheduler_type cosine
    --warmup_steps 250
    --weight_decay 1e-6
    --logging_steps 20
    --eval_steps 50

    # Miscellaneous Settings (unlikely to change)
    --do_train
    --stage sft
    --template qwen_sd
    --dataset_dir data
    --bf16
    --preprocessing_num_workers 20
    --ignore_pad_token_for_loss True
    --overwrite_output_dir
    --overwrite_cache
    --evaluation_strategy steps
    --save_strategy steps
    --deepspeed x_ds_z3_config.json
)

# Run the command with all arguments
"${args[@]}"
