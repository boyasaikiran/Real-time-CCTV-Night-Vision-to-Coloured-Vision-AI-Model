from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image
from torch.utils.data import Dataset


class LLVIPDataset(Dataset):

    def __init__(self,
                 dataset_path,
                 split="train",
                 transform=None):

        self.dataset_path = Path(dataset_path)

        self.transform = transform

        self.split = split

        self.ir_folder = self.dataset_path / "infrared" / split

        self.rgb_folder = self.dataset_path / "visible" / split

        self.annotation_folder = self.dataset_path / "Annotations"

        self.ir_images = sorted(self.ir_folder.glob("*.jpg"))

        self.rgb_images = sorted(self.rgb_folder.glob("*.jpg"))

        assert len(self.ir_images) == len(self.rgb_images), \
            "Infrared and Visible images are not paired"

    def __len__(self):

        return len(self.ir_images)

    def load_annotation(self, image_name):

        xml_file = self.annotation_folder / f"{image_name}.xml"

        if not xml_file.exists():
            return []

        tree = ET.parse(xml_file)

        root = tree.getroot()

        objects = []

        for obj in root.findall("object"):

            name = obj.find("name").text

            bbox = obj.find("bndbox")

            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            objects.append({

                "label": name,

                "bbox": [xmin, ymin, xmax, ymax]

            })

        return objects

    def __getitem__(self, index):

        ir_path = self.ir_images[index]

        rgb_path = self.rgb_images[index]

        ir_image = Image.open(ir_path).convert("RGB")

        rgb_image = Image.open(rgb_path).convert("RGB")

        annotation = self.load_annotation(ir_path.stem)

        if self.transform:

            ir_image = self.transform(ir_image)

            rgb_image = self.transform(rgb_image)

        sample = {

            "infrared": ir_image,

            "visible": rgb_image,

            "annotation": annotation,

            "filename": ir_path.name

        }

        return sample