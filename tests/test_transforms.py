from datasets.dataset import LLVIPDataset
from datasets.transforms import LLVIPTransforms

transform = LLVIPTransforms().train_transform()

dataset = LLVIPDataset(
    dataset_path="/kaggle/input/datasets/boyasaikiran/llvip-dataset/LLVIP",
    split="train",
    transform=transform
)

sample = dataset[0]

print("=" * 60)
print("TRANSFORM TEST")
print("=" * 60)

print()

print("Infrared Shape")

print(sample["infrared"].shape)

print()

print("Visible Shape")

print(sample["visible"].shape)

print()

print("Filename")

print(sample["filename"])

print()

print("Objects")

print(len(sample["annotation"]))