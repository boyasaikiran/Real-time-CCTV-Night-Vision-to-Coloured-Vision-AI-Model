import torch

from models.colorization.discriminator import Discriminator

model = Discriminator()

infrared = torch.randn(1,3,256,256)

visible = torch.randn(1,3,256,256)

print("="*60)
print("PATCHGAN TEST")
print("="*60)

print("\nInfrared")

print(infrared.shape)

print("\nVisible")

print(visible.shape)

output = model(
    infrared,
    visible
)

print("\nPatchGAN Output")

print(output.shape)

print("\nDiscriminator Working Successfully")