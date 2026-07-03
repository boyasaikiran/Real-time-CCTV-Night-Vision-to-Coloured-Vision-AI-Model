"""
validator.py

Phase 1 - Dataset Engineering

Project:
Real-Time CCTV Night Vision to Coloured Vision AI Model

Author:
Boya Sai Kiran
"""

import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm


class LLVIPValidator:

    def __init__(self, dataset_path):

        self.dataset_path = Path(dataset_path)

        self.ir_train = self.dataset_path / "infrared" / "train"
        self.ir_test = self.dataset_path / "infrared" / "test"

        self.rgb_train = self.dataset_path / "visible" / "train"
        self.rgb_test = self.dataset_path / "visible" / "test"

        self.annotation_path = self.dataset_path / "Annotations"

    def count_images(self, folder):

        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

        images = []

        for ext in image_extensions:
            images.extend(folder.glob(f"*{ext}"))

        return images

    def validate_pairs(self):

        print("=" * 60)
        print("Checking Infrared ↔ Visible Image Pairs")
        print("=" * 60)

        ir_train = self.count_images(self.ir_train)
        rgb_train = self.count_images(self.rgb_train)

        ir_test = self.count_images(self.ir_test)
        rgb_test = self.count_images(self.rgb_test)

        print(f"Infrared Train Images : {len(ir_train)}")
        print(f"Visible Train Images   : {len(rgb_train)}")

        print()

        print(f"Infrared Test Images : {len(ir_test)}")
        print(f"Visible Test Images  : {len(rgb_test)}")

        print()

        if len(ir_train) == len(rgb_train):
            print("✅ Train images are paired correctly.")
        else:
            print("❌ Train images are NOT paired.")

        if len(ir_test) == len(rgb_test):
            print("✅ Test images are paired correctly.")
        else:
            print("❌ Test images are NOT paired.")

    def validate_annotations(self):

        xml_files = list(self.annotation_path.glob("*.xml"))

        print()
        print("=" * 60)
        print("Checking Annotation Files")
        print("=" * 60)

        print(f"Annotation Files : {len(xml_files)}")

    def validate_images(self):

        print()
        print("=" * 60)
        print("Checking Image Integrity")
        print("=" * 60)

        folders = [
            self.ir_train,
            self.ir_test,
            self.rgb_train,
            self.rgb_test,
        ]

        corrupted = 0

        for folder in folders:

            images = self.count_images(folder)

            for image in tqdm(images):

                try:

                    img = Image.open(image)
                    img.verify()

                except Exception:

                    corrupted += 1
                    print(f"Corrupted : {image}")

        print()

        if corrupted == 0:
            print("✅ No corrupted images found.")
        else:
            print(f"❌ Corrupted Images : {corrupted}")

    def run(self):

        print()
        print("=" * 60)
        print("LLVIP DATASET VALIDATION")
        print("=" * 60)

        self.validate_pairs()

        self.validate_annotations()

        self.validate_images()

        print()
        print("=" * 60)
        print("Dataset Validation Completed")
        print("=" * 60)


if __name__ == "__main__":

    dataset_path = "datasets/raw/LLVIP"

    validator = LLVIPValidator(dataset_path)

    validator.run()