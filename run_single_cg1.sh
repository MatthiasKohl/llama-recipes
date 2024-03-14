#!/usr/bin/env bash

# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

OMP_NUM_THREADS=4 python \
    examples/finetuning.py \
    --batch_size_training 2 \
    --fsdp_config.interleaved_offload_param 2 \
    --fsdp_config.interleaved_offload_act 2 \
    --dist_checkpoint_root_folder model_checkpoints \
    --dist_checkpoint_folder fine-tuned \
    --model_name meta-llama/Llama-2-70b-hf \
    --use_fast_kernels \
    --fsdp_config.pure_bf16 \
    --use_peft \
    --gradient_accumulation_steps 20 \
    --max_steps_per_epoch 160
