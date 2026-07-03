"""
collate.py

Custom collate function for LLVIP Dataset
"""

import torch


def llvip_collate_fn(batch):
    """
    Custom collate function for LLVIP object detection dataset.
    Images are stacked into tensors.
    Annotations remain as Python lists because each image
    may contain a different number of objects.
    """

    infrared = torch.stack([sample["infrared"] for sample in batch], dim=0)

    visible = torch.stack([sample["visible"] for sample in batch], dim=0)

    annotations = [sample["annotation"] for sample in batch]

    filenames = [sample["filename"] for sample in batch]

    return {
        "infrared": infrared,
        "visible": visible,
        "annotation": annotations,
        "filename": filenames,
    }