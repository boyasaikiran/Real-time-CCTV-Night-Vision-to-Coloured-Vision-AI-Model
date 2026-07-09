"""
paths.py

Project Paths Configuration
"""

import os

# ============================================================
# Dataset Path
# ============================================================

if os.path.exists("/kaggle/input"):

    DATASET_PATH = "/kaggle/input/datasets/boyasaikiran/llvip-dataset/LLVIP"

else:

    DATASET_PATH = "datasets/raw/LLVIP"