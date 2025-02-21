#!/bin/bash

# Disable version check
export DISABLE_VERSION_CHECK=1

# Define arguments using an array for readability
args=(

    # General Torchrun Settings
    python ./src/train.py

    # Model Settings
    --run_name slider_test
    --model_name_or_path "Qwen/Qwen2.5-1.5B-Instruct"
    --finetuning_type lora
    --lora_target all
    --lora_rank 4
    --lora_dropout 0.1
    --slider_on True
    --slider_n_variables 3
    --slider_n_hidden 128
    --slider_n_heads_sharing_slider 2
    --slider_dropout 0.1
    --slider_attn_factor 1.0

    # Data Settings
    --dataset slider_test
    --val_size 0.02
    --cutoff_len 4096

    # Output Settings
    --output_dir slider_test_output
    --report_to none
    --save_steps 10
    --save_total_limit 2

    # Training Hyperparameters
    --num_train_epochs 2
    --per_device_train_batch_size 2
    --per_device_eval_batch_size 2
    --gradient_accumulation_steps 2
    --learning_rate 1e-4
    --lr_scheduler_type cosine
    --warmup_steps 10
    --weight_decay 1e-6
    --logging_steps 5
    --eval_steps 10

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
)

# Run the command with all arguments
"${args[@]}"
