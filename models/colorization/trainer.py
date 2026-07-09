"""
trainer.py

Pix2Pix Trainer
"""

import os
import torch
import torchvision.utils as vutils
from tqdm import tqdm


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        optimizer_G,
        optimizer_D,
        device
    ):

        self.model = model.to(device)
        self.train_loader = train_loader
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.device = device

        # ============================================================
        # Output Directory
        # ============================================================

        self.output_dir = "outputs"
        os.makedirs(self.output_dir, exist_ok=True)

    # ============================================================
    # Save Sample Images
    # ============================================================

    def save_sample(self, infrared, visible, epoch):

        self.model.eval()

        with torch.no_grad():
            generated = self.model.generate(infrared)

        # Convert from [-1,1] → [0,1]
        infrared = infrared.cpu() * 0.5 + 0.5
        visible = visible.cpu() * 0.5 + 0.5
        generated = generated.cpu() * 0.5 + 0.5

        # ------------------------------------------------------------
        # Save Generated Image
        # ------------------------------------------------------------

        generated_file = os.path.join(
            self.output_dir,
            f"epoch_{epoch:03d}_generated.png"
        )

        vutils.save_image(
            generated,
            generated_file
        )

        # ------------------------------------------------------------
        # Save Comparison Image
        # ------------------------------------------------------------

        comparison = torch.cat(
            [
                infrared,
                generated,
                visible
            ],
            dim=3
        )

        comparison_file = os.path.join(
            self.output_dir,
            f"epoch_{epoch:03d}_comparison.png"
        )

        vutils.save_image(
            comparison,
            comparison_file
        )

        print()
        print(f"Generated Image  : {generated_file}")
        print(f"Comparison Image : {comparison_file}")

        self.model.train()

    # ============================================================
    # Train One Epoch
    # ============================================================

    def train_epoch(self, epoch):

        self.model.train()

        running_g = 0.0
        running_d = 0.0

        progress = tqdm(self.train_loader)

        sample_infrared = None
        sample_visible = None

        for batch in progress:

            infrared = batch["infrared"].to(self.device)
            visible = batch["visible"].to(self.device)

            # Save first sample of epoch
            if sample_infrared is None:
                sample_infrared = infrared[:1].clone()
                sample_visible = visible[:1].clone()

            ############################################################
            # Generator
            ############################################################

            fake = self.model.generate(infrared)

            fake_prediction = self.model.discriminate(
                infrared,
                fake
            )

            generator_loss = self.model.generator_loss(
                fake_prediction,
                fake,
                visible
            )

            self.optimizer_G.zero_grad()

            generator_loss["total"].backward()

            self.optimizer_G.step()

            ############################################################
            # Discriminator
            ############################################################

            real_prediction = self.model.discriminate(
                infrared,
                visible
            )

            fake_prediction = self.model.discriminate(
                infrared,
                fake.detach()
            )

            discriminator_loss = self.model.discriminator_loss(
                real_prediction,
                fake_prediction
            )

            self.optimizer_D.zero_grad()

            discriminator_loss.backward()

            self.optimizer_D.step()

            running_g += generator_loss["total"].item()
            running_d += discriminator_loss.item()

            progress.set_postfix(
                G_Loss=f"{generator_loss['total'].item():.4f}",
                D_Loss=f"{discriminator_loss.item():.4f}"
            )

        ############################################################
        # Save Sample
        ############################################################

        self.save_sample(
            sample_infrared,
            sample_visible,
            epoch
        )

        ############################################################
        # Return Average Loss
        ############################################################

        return (
            running_g / len(self.train_loader),
            running_d / len(self.train_loader)
        )