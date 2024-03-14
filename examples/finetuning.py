# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

try:
    import fire
except ImportError:
    import shutil
    tr_path = shutil.which('torchrun')
    py_path = shutil.which('python')
    print(
        f"Importing fire failed. You may need to update torchrun script "
        f"to use {py_path} with: sudo nano {tr_path}"
    )
    raise
from llama_recipes.finetuning import main

if __name__ == "__main__":
    fire.Fire(main)
