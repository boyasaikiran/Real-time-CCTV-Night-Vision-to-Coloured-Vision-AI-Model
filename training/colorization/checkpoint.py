"""
checkpoint.py

Save and Load Model Checkpoints
"""

import os
import torch


class CheckpointManager:

    def __init__(self, save_dir="weights/colorization"):

        self.save_dir = save_dir

        os.makedirs(save_dir, exist_ok=True)

    def save(
        self,
        epoch,
        generator,
        discriminator,
        optimizer_G,
        optimizer_D
    ):

        checkpoint = {

            "epoch": epoch,

            "generator": generator.state_dict(),

            "discriminator": discriminator.state_dict(),

            "optimizer_G": optimizer_G.state_dict(),

            "optimizer_D": optimizer_D.state_dict()

        }

        filename = os.path.join(
            self.save_dir,
            f"checkpoint_epoch_{epoch}.pth"
        )

        torch.save(checkpoint, filename)

        print(f"\nCheckpoint Saved : {filename}")

    def load(
        self,
        filename,
        generator,
        discriminator,
        optimizer_G=None,
        optimizer_D=None
    ):

        checkpoint = torch.load(
            filename,
            map_location="cpu"
        )

        generator.load_state_dict(
            checkpoint["generator"]
        )

        discriminator.load_state_dict(
            checkpoint["discriminator"]
        )

        if optimizer_G:

            optimizer_G.load_state_dict(
                checkpoint["optimizer_G"]
            )

        if optimizer_D:

            optimizer_D.load_state_dict(
                checkpoint["optimizer_D"]
            )

        print("\nCheckpoint Loaded Successfully")

        return checkpoint["epoch"]