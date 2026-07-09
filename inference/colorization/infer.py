"""
infer.py

Run inference using trained Pix2Pix Generator
"""

import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils

from models.colorization.generator import Generator


# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("DEVICE")
print("=" * 60)
print(DEVICE)
print()


# ============================================================
# Paths
# ============================================================

CHECKPOINT = "weights/colorization/checkpoint_epoch_1.pth"

INPUT_IMAGE = "datasets/raw/LLVIP/infrared/test/260536.jpg"

OUTPUT_DIR = "outputs"

OUTPUT_IMAGE = os.path.join(
    OUTPUT_DIR,
    "generated.png"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Verify Files
# ============================================================

if not os.path.exists(CHECKPOINT):
    raise FileNotFoundError(
        f"Checkpoint not found:\n{CHECKPOINT}"
    )

if not os.path.exists(INPUT_IMAGE):
    raise FileNotFoundError(
        f"Input image not found:\n{INPUT_IMAGE}"
    )


# ============================================================
# Transform
# ============================================================

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])


# ============================================================
# Load Generator
# ============================================================

print("=" * 60)
print("LOADING MODEL")
print("=" * 60)

generator = Generator().to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

# ------------------------------------------------------------
# Support multiple checkpoint formats
# ------------------------------------------------------------

if isinstance(checkpoint, dict):

    if "generator_state_dict" in checkpoint:

        generator.load_state_dict(
            checkpoint["generator_state_dict"]
        )

    elif "generator" in checkpoint:

        generator.load_state_dict(
            checkpoint["generator"]
        )

    else:

        generator.load_state_dict(checkpoint)

else:

    generator.load_state_dict(checkpoint)

generator.eval()

print("Checkpoint Loaded Successfully")
print()


# ============================================================
# Load Image
# ============================================================

print("=" * 60)
print("READING IMAGE")
print("=" * 60)

image = Image.open(INPUT_IMAGE).convert("RGB")

image = transform(image)

image = image.unsqueeze(0).to(DEVICE)


# ============================================================
# Inference
# ============================================================

print("Generating Image...")
print()

with torch.no_grad():

    output = generator(image)

output = output.squeeze(0)

output = output.cpu()

# Convert [-1,1] -> [0,1]
output = output * 0.5 + 0.5


# ============================================================
# Save Image
# ============================================================

vutils.save_image(
    output,
    OUTPUT_IMAGE
)

print("=" * 60)
print("INFERENCE COMPLETED")
print("=" * 60)
print()

print("Saved Output:")
print(OUTPUT_IMAGE)
print()