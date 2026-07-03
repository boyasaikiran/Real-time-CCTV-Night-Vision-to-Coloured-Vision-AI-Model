import torch

from models.colorization.pix2pix import Pix2Pix

from training.colorization.trainer import Trainer

print("="*60)
print("TRAINER TEST")
print("="*60)

model = Pix2Pix()

optimizer_G = torch.optim.Adam(
    model.generator.parameters(),
    lr=0.0002
)

optimizer_D = torch.optim.Adam(
    model.discriminator.parameters(),
    lr=0.0002
)

print()

print("Generator Parameters")

print(sum(
    p.numel()
    for p in model.generator.parameters()
))

print()

print("Discriminator Parameters")

print(sum(
    p.numel()
    for p in model.discriminator.parameters()
))

print()

print("Trainer Ready")