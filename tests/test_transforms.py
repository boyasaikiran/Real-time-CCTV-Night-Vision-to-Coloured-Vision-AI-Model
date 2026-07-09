"""
Test Transform Pipeline
"""

import os

from datasets.dataset import LLVIPDataset
from datasets.transforms import LLVIPTransforms


# ============================================================
# Dataset Path
# ============================================================

from configs.paths import DATASET_PATH


# ============================================================
# Transform
# ============================================================

transform = LLVIPTransforms().train_transform()


# ============================================================
# Dataset
# ============================================================

dataset = LLVIPDataset(
    dataset_path=DATASET_PATH,
    split="train",
    transform=transform
)

sample = dataset[0]


# ============================================================
# Output
# ============================================================

print("=" * 60)
print("TRANSFORM TEST")
print("=" * 60)

print()

print("Infrared Shape")
print(sample["infrared"].shape)

print()

print("Visible Shape")
print(sample["visible"].shape)

print()

print("Filename")
print(sample["filename"])

print()

print("Objects")
print(len(sample["annotation"]))