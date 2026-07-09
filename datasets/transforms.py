"""
transforms.py

Image Transformation Pipeline for LLVIP Dataset
Research Baseline Version
"""

from torchvision import transforms


class LLVIPTransforms:

    def __init__(self, image_size=(512, 512)):
        self.image_size = image_size

    def train_transform(self):

        return transforms.Compose([

            transforms.Resize(self.image_size),

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