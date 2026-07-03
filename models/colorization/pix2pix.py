"""
pix2pix.py

Complete Pix2Pix Model
"""

import torch.nn as nn

from models.colorization.generator import Generator
from models.colorization.discriminator import Discriminator
from models.colorization.losses import Pix2PixLoss


class Pix2Pix(nn.Module):

    def __init__(self):

        super().__init__()

        self.generator = Generator()

        self.discriminator = Discriminator()

        self.loss_fn = Pix2PixLoss()

    def generate(self, infrared):

        return self.generator(infrared)

    def discriminate(self, infrared, visible):

        return self.discriminator(infrared, visible)

    def generator_loss(
        self,
        fake_prediction,
        fake_image,
        real_image
    ):

        return self.loss_fn.generator_loss(
            fake_prediction,
            fake_image,
            real_image
        )

    def discriminator_loss(
        self,
        real_prediction,
        fake_prediction
    ):

        return self.loss_fn.discriminator_loss(
            real_prediction,
            fake_prediction
        )