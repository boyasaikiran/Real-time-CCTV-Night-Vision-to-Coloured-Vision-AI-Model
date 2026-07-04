from datasets.dataset import LLVIPDataset

dataset = LLVIPDataset(

    dataset_path="/kaggle/input/datasets/boyasaikiran/llvip-dataset/LLVIP",

    split="train"

)

print()

print("=" * 60)

print("DATASET INFORMATION")

print("=" * 60)

print("Dataset Size :", len(dataset))

sample = dataset[0]

print()

print("Keys")

print(sample.keys())

print()

print("Filename")

print(sample["filename"])

print()

print("Annotation")

print(sample["annotation"][:3])

print()

print("Infrared Size")

print(sample["infrared"].size)

print()

print("Visible Size")

print(sample["visible"].size)