from datasets.loader import LLVIPDataLoader

loader = LLVIPDataLoader(
    dataset_path="datasets/raw/LLVIP",
    batch_size=4,
)

train_loader = loader.train_loader()

print("=" * 60)
print("DATALOADER TEST")
print("=" * 60)

print("\nNumber of Batches:")
print(len(train_loader))

batch = next(iter(train_loader))

print("\nKeys:")
print(batch.keys())

print("\nInfrared Batch Shape:")
print(batch["infrared"].shape)

print("\nVisible Batch Shape:")
print(batch["visible"].shape)

print("\nImages in Batch:")
print(len(batch["annotation"]))

print("\nFirst Image Object Count:")
print(len(batch["annotation"][0]))

print("\nFilenames:")
print(batch["filename"])