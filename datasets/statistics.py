"""
statistics.py

Generate statistics for the LLVIP dataset.
"""

from pathlib import Path
import xml.etree.ElementTree as ET
from collections import Counter
from PIL import Image


class LLVIPStatistics:

    def __init__(self, dataset_path):

        self.dataset_path = Path(dataset_path)

        self.ir_train = self.dataset_path / "infrared" / "train"
        self.ir_test = self.dataset_path / "infrared" / "test"

        self.annotation_path = self.dataset_path / "Annotations"

    def count_images(self):

        train_images = list(self.ir_train.glob("*.jpg"))
        test_images = list(self.ir_test.glob("*.jpg"))

        return len(train_images), len(test_images)

    def average_resolution(self):

        widths = []
        heights = []

        for image in self.ir_train.glob("*.jpg"):

            img = Image.open(image)

            w, h = img.size

            widths.append(w)
            heights.append(h)

        avg_w = sum(widths) / len(widths)
        avg_h = sum(heights) / len(heights)

        return avg_w, avg_h

    def annotation_statistics(self):

        counter = Counter()

        total_objects = 0

        xml_files = list(self.annotation_path.glob("*.xml"))

        for xml in xml_files:

            tree = ET.parse(xml)

            root = tree.getroot()

            for obj in root.findall("object"):

                label = obj.find("name").text

                counter[label] += 1

                total_objects += 1

        return counter, total_objects

    def report(self):

        train_count, test_count = self.count_images()

        avg_w, avg_h = self.average_resolution()

        counter, total_objects = self.annotation_statistics()

        print("=" * 60)
        print("LLVIP DATASET STATISTICS")
        print("=" * 60)

        print(f"\nTrain Images : {train_count}")
        print(f"Test Images  : {test_count}")

        print(f"\nAverage Resolution : {avg_w:.0f} x {avg_h:.0f}")

        print(f"\nTotal Objects : {total_objects}")

        print("\nClass Distribution")

        for cls, num in counter.items():

            print(f"{cls:15} {num}")


if __name__ == "__main__":

    stats = LLVIPStatistics("datasets/raw/LLVIP")

    stats.report()