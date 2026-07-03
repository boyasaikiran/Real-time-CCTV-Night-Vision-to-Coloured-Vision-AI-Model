"""
preprocess.py

Preprocess LLVIP Dataset

- Resize Images
- Save Processed Dataset
"""

from pathlib import Path

from PIL import Image
from tqdm import tqdm


class LLVIPPreprocessor:

    def __init__(self,
                 raw_dataset,
                 output_dataset,
                 image_size=(512, 512)):

        self.raw = Path(raw_dataset)

        self.output = Path(output_dataset)

        self.image_size = image_size

    def process_folder(self, input_folder, output_folder):

        output_folder.mkdir(parents=True, exist_ok=True)

        images = list(input_folder.glob("*.jpg"))

        for image in tqdm(images):

            img = Image.open(image).convert("RGB")

            img = img.resize(self.image_size)

            save_path = output_folder / image.name

            img.save(save_path)

    def process(self):

        folders = [

            ("infrared/train", "infrared/train"),

            ("infrared/test", "infrared/test"),

            ("visible/train", "visible/train"),

            ("visible/test", "visible/test"),

        ]

        print("=" * 60)
        print("LLVIP PREPROCESSING")
        print("=" * 60)

        for input_dir, output_dir in folders:

            print(f"\nProcessing {input_dir}")

            self.process_folder(

                self.raw / input_dir,

                self.output / output_dir

            )

        print()

        print("=" * 60)
        print("PREPROCESSING COMPLETED")
        print("=" * 60)


if __name__ == "__main__":

    processor = LLVIPPreprocessor(

        raw_dataset="datasets/raw/LLVIP",

        output_dataset="datasets/processed",

        image_size=(512, 512)

    )

    processor.process()