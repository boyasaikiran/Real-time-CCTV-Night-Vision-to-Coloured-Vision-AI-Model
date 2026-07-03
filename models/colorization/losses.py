"""
losses.py

Pix2Pix Loss Functions
"""

import torch
import torch.nn as nn
import torchvision.models as models


class GANLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, prediction, target):
        return self.loss(prediction, target)


class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, fake, real):
        return self.loss(fake, real)


class PerceptualLoss(nn.Module):

    def __init__(self):

        super().__init__()

        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)

        self.features = vgg.features[:35].eval()

        for p in self.features.parameters():
            p.requires_grad = False

        self.loss = nn.L1Loss()

    def forward(self, fake, real):

        fake_features = self.features(fake)

        real_features = self.features(real)

        return self.loss(fake_features, real_features)


class Pix2PixLoss(nn.Module):

    def __init__(self, lambda_l1=100):

        super().__init__()

        self.gan = GANLoss()

        self.l1 = L1Loss()

        self.perceptual = PerceptualLoss()

        self.lambda_l1 = lambda_l1

    def generator_loss(
        self,
        fake_prediction,
        fake_image,
        real_image
    ):

        gan_loss = self.gan(
            fake_prediction,
            torch.ones_like(fake_prediction)
        )

        l1_loss = self.l1(
            fake_image,
            real_image
        )

        perceptual_loss = self.perceptual(
            fake_image,
            real_image
        )

        total = (
            gan_loss +
            self.lambda_l1 * l1_loss +
            perceptual_loss
        )

        return {
            "total": total,
            "gan": gan_loss,
            "l1": l1_loss,
            "perceptual": perceptual_loss
        }

    def discriminator_loss(
        self,
        real_prediction,
        fake_prediction
    ):

        real_loss = self.gan(
            real_prediction,
            torch.ones_like(real_prediction)
        )

        fake_loss = self.gan(
            fake_prediction,
            torch.zeros_like(fake_prediction)
        )

        return (real_loss + fake_loss) * 0.5