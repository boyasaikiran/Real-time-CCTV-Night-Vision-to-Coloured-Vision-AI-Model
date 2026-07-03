import torch

from models.colorization.pix2pix import Pix2Pix

from training.colorization.checkpoint import CheckpointManager

print("="*60)
print("CHECKPOINT TEST")
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

manager = CheckpointManager()

manager.save(

    epoch=1,

    generator=model.generator,

    discriminator=model.discriminator,

    optimizer_G=optimizer_G,

    optimizer_D=optimizer_D

)

print()

print("Checkpoint Test Passed")