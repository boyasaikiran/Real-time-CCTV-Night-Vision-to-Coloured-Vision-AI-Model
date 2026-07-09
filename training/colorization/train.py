"""
train.py

Train Pix2Pix on LLVIP
"""

import os
import torch

from datasets.loader import LLVIPDataLoader
from models.colorization.pix2pix import Pix2Pix
from training.colorization.trainer import Trainer
from training.colorization.checkpoint import CheckpointManager


# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("DEVICE")
print("=" * 60)
print(DEVICE)
print()


# ============================================================
# Dataset Path (Automatic Windows / Kaggle Detection)
# ============================================================

from configs.paths import DATASET_PATH


# ============================================================
# Dataset
# ============================================================

loader = LLVIPDataLoader(
    dataset_path=DATASET_PATH,
    batch_size=2,
    num_workers=2
)

train_loader = loader.train_loader()


# ============================================================
# Model
# ============================================================

model = Pix2Pix()


# ============================================================
# Optimizers
# ============================================================

optimizer_G = torch.optim.Adam(
    model.generator.parameters(),
    lr=0.0002,
    betas=(0.5, 0.999)
)

optimizer_D = torch.optim.Adam(
    model.discriminator.parameters(),
    lr=0.0002,
    betas=(0.5, 0.999)
)


# ============================================================
# Trainer
# ============================================================

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    optimizer_G=optimizer_G,
    optimizer_D=optimizer_D,
    device=DEVICE
)


# ============================================================
# Checkpoint Manager
# ============================================================

checkpoint = CheckpointManager()


# ============================================================
# Training Configuration
# ============================================================

EPOCHS = 25


# ============================================================
# Training Loop
# ============================================================

print("=" * 60)
print("STARTING TRAINING")
print("=" * 60)

for epoch in range(EPOCHS):

    print()
    print("=" * 60)
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print("=" * 60)

    g_loss, d_loss = trainer.train_epoch()

    print()
    print(f"Generator Loss     : {g_loss:.4f}")
    print(f"Discriminator Loss : {d_loss:.4f}")

    print(r"""

███████╗ █████╗ ██╗
██╔════╝██╔══██╗██║
███████╗███████║██║
╚════██║██╔══██║██║
███████║██║  ██║██║
╚══════╝╚═╝  ╚═╝╚═╝

""")

    checkpoint.save(
        epoch=epoch + 1,
        generator=model.generator,
        discriminator=model.discriminator,
        optimizer_G=optimizer_G,
        optimizer_D=optimizer_D
    )

print()
print("=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)