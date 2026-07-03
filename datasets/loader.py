"""
loader.py

LLVIP DataLoader Module
"""

from torch.utils.data import DataLoader

from datasets.dataset import LLVIPDataset
from datasets.transforms import LLVIPTransforms
from datasets.collate import llvip_collate_fn


class LLVIPDataLoader:

    def __init__(
        self,
        dataset_path,
        image_size=(512, 512),
        batch_size=4,
        num_workers=0,
    ):

        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transforms = LLVIPTransforms(image_size)

    def train_loader(self):

        dataset = LLVIPDataset(
            dataset_path=self.dataset_path,
            split="train",
            transform=self.transforms.train_transform(),
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=llvip_collate_fn,
        )

    def test_loader(self):

        dataset = LLVIPDataset(
            dataset_path=self.dataset_path,
            split="test",
            transform=self.transforms.test_transform(),
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=llvip_collate_fn,
        )