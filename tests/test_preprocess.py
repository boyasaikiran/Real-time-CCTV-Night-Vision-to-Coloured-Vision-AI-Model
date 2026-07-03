from PIL import Image
from pathlib import Path

image = Path(
    "datasets/processed/infrared/train"
).glob("*.jpg")

image = next(image)

img = Image.open(image)

print("=" * 60)

print("PREPROCESS TEST")

print("=" * 60)

print()

print("Image")

print(image.name)

print()

print("Size")

print(img.size)