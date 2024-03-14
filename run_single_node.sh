#!/usr/bin/env bash

# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

OMP_NUM_THREADS=4 torchrun --nnodes 1 --nproc_per_node 8 \
    examples/finetuning.py \
    --enable_fsdp \
    --low_cpu_fsdp \
    --batch_size_training 4 \
    --dist_checkpoint_root_folder model_checkpoints \
    --dist_checkpoint_folder fine-tuned \
    --model_name meta-llama/Llama-2-70b-hf \
    --use_fast_kernels \
    --fsdp_config.pure_bf16 \
    --fsdp_config.fsdp_activation_checkpointing \
    --use_peft
    # --apply_optim_backward
