import torch

from models.colorization.losses import Pix2PixLoss

loss_fn = Pix2PixLoss()

fake = torch.randn(1,3,256,256)

real = torch.randn(1,3,256,256)

prediction = torch.randn(1,1,15,15)

print("="*60)
print("LOSS TEST")
print("="*60)

loss = loss_fn.generator_loss(
    prediction,
    fake,
    real
)

print()

for key,value in loss.items():
    print(key,":",value.item())

print()

disc = loss_fn.discriminator_loss(
    prediction,
    prediction
)

print("Discriminator Loss :",disc.item())

print()

print("Loss Module Working Successfully")