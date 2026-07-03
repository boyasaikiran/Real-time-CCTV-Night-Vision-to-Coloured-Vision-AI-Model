"""
transforms.py

Image transformation pipeline for LLVIP Dataset
"""

from torchvision import transforms


class LLVIPTransforms:

    def __init__(self, image_size=(512, 512)):

        self.image_size = image_size

    def train_transform(self):

        return transforms.Compose([

            transforms.Resize(self.image_size),

            transforms.RandomHorizontalFlip(p=0.5),

            transforms.RandomRotation(5),

            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )

        ])

    def test_transform(self):

        return transforms.Compose([

            transforms.Resize(self.image_size),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )

        ])