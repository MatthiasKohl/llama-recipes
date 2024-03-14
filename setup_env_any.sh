#!/usr/bin/env bash

# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

# we expect this to be called through "source" to setup the calling shell's environment

set -x
trap 'trap - ERR RETURN; set +x; kill -INT $$ ; return' ERR RETURN

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"

export HF_HOME=$SCRIPT_DIR/hf_home
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
