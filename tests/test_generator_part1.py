import torch

from models.colorization.generator import Generator

model = Generator()

x = torch.randn(1,3,256,256)

print("="*60)
print("GENERATOR TEST")
print("="*60)

print("Input :",x.shape)

y=model(x)

print("Output:",y.shape)

print("\nGenerator Working Successfully")