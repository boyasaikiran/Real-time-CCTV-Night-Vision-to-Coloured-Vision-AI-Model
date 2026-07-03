"""
trainer.py

Pix2Pix Trainer
"""

import torch
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

    def train_epoch(self):

        self.model.train()

        running_g = 0.0
        running_d = 0.0

        progress = tqdm(self.train_loader)

        for batch in progress:

            infrared = batch["infrared"].to(self.device)

            visible = batch["visible"].to(self.device)

            ##############################
            # Generator
            ##############################

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

            ##############################
            # Discriminator
            ##############################

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

                G_Loss=generator_loss["total"].item(),

                D_Loss=discriminator_loss.item()

            )

        return (

            running_g / len(self.train_loader),

            running_d / len(self.train_loader)

        )