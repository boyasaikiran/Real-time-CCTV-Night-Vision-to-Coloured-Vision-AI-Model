import torch

from models.colorization.pix2pix import Pix2Pix

model = Pix2Pix()

infrared = torch.randn(1,3,256,256)

visible = torch.randn(1,3,256,256)

print("="*60)
print("PIX2PIX TEST")
print("="*60)

fake = model.generate(infrared)

print()

print("Generated Image")

print(fake.shape)

real_pred = model.discriminate(
    infrared,
    visible
)

fake_pred = model.discriminate(
    infrared,
    fake.detach()
)

print()

print("Real Prediction")

print(real_pred.shape)

print()

print("Fake Prediction")

print(fake_pred.shape)

g_loss = model.generator_loss(
    fake_pred,
    fake,
    visible
)

d_loss = model.discriminator_loss(
    real_pred,
    fake_pred
)

print()

print("Generator Total Loss")

print(g_loss["total"].item())

print()

print("Discriminator Loss")

print(d_loss.item())

print()

print("PIX2PIX MODEL WORKING SUCCESSFULLY")