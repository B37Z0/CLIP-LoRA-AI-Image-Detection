"""
Tiny-GenImage data pipeline.

- Load TheKernel01/Tiny-GenImage from HuggingFace
- Provide cross-generator train/test split -> generalization
  to unseen generators is the project's primary interest

Python 3.15 has a cool lazy import I would've liked to use
instead of the deferred imports but this is fine... 
"""

from typing import Callable, Optional
from PIL import Image
from torch.utils.data import Dataset, Subset

# Tiny-GenImage generator class labels (different from full GenImage)
GENERATOR_NAMES = ["Real", "ADM", "BigGAN", "GLIDE", "Midjourney", "SD14", "SD15", "VQDM", "Wukong"]


class TinyGenImageDataset(Dataset):
    def __init__(self, split: str = "train", transform: Optional[Callable] = None):
        from datasets import load_dataset # deferred import 1
        assert split in ("train", "validation")
        self.hf_dataset = load_dataset("TheKernel01/Tiny-GenImage", split=split)
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        row = self.hf_dataset[idx]
        image = row["image"].convert("RGB") # normalize to 3-channel RGB
        label = row["label"]                # 0 = real, 1 = fake
        generator = GENERATOR_NAMES[row["generator"]]
        if self.transform is not None:
            image = self.transform(image)

        # Add generator name for logging per-generator metrics
        return image, label, generator

    def get_generator(self, idx):
        """Return generator name for sample `idx` without decoding the image.
        HuggingFace `datasets` lazily decodes images, so skipping the image
        column avoids the JPEG decode cost entirely."""
        return GENERATOR_NAMES[self.hf_dataset[idx]["generator"]]

    def get_label(self, idx):
        """Return real/fake label for sample `idx` without decoding the
        image — same rationale as get_generator."""
        return self.hf_dataset[idx]["label"]


class CrossGeneratorSplit:
    """
    Split TinyGenImageDataset so `held_out_generators` don't appear in
    the train set. Real images are included in both the train and test set.
    """

    def __init__(self, dataset: TinyGenImageDataset, held_out_generators: list):
        self.held_out = set(held_out_generators)

        train_idx, test_idx = [], []
        for i in range(len(dataset)):
            generator = dataset.get_generator(i)
            if generator == "Real":
                train_idx.append(i)
                test_idx.append(i)
            elif generator in self.held_out:
                test_idx.append(i)
            else:
                train_idx.append(i)

        self.train_indices = train_idx
        self.test_indices = test_idx
        self.dataset = dataset

    def get_splits(self):
        return Subset(self.dataset, self.train_indices), Subset(self.dataset, self.test_indices)


def build_clip_transform(model_id: str = "openai/clip-vit-large-patch14"):
    """
    Build a transform that preprocesses PIL images exactly as CLIP expects.
    No need to code myself + transform stays correct even if the model is 
    swapped out.
    """
    from transformers import CLIPImageProcessor # deferred import 2
    processor = CLIPImageProcessor.from_pretrained(model_id)

    def transform(image: Image.Image):
        # processor() returns a BatchFeature dict; pixel_values is the key
        # and contains processed image tensors. [0] unwraps the added batch
        # dimension so the DataLoader can rebatch.
        return processor(images=image, return_tensors="pt")["pixel_values"][0]

    return transform


if __name__ == "__main__":
    transform = build_clip_transform()

    ds = TinyGenImageDataset(split="train", transform=transform)
    print(f"Total train samples: {len(ds)}")

    held_out = ["Midjourney", "VQDM"]
    splitter = CrossGeneratorSplit(ds, held_out_generators=held_out)
    train_set, test_set = splitter.get_splits()
    print(f"Train (excluding {held_out}): {len(train_set)} samples")
    print(f"Test (only {held_out} + real): {len(test_set)} samples")

    train_generators = {ds.get_generator(i) for i in splitter.train_indices}

    # Just in case
    overlap = train_generators & set(held_out)
    assert not overlap, f"held-out generators leaked into train set: {overlap}"
    print("Cross-generator split verified.")

    # Check real/fake balance per split since held-out test sets 
    # might be dominated by real images after most of the generators' 
    # fakes were excluded
    def label_counts(indices):
        real = sum(1 for i in indices if ds.get_label(i) == 0)
        return real, len(indices) - real

    train_real, train_fake = label_counts(splitter.train_indices)
    test_real, test_fake = label_counts(splitter.test_indices)
    print(f"Train balance: {train_real} real / {train_fake} fake "
          f"({100 * train_real / (train_real + train_fake):.1f}% real)")
    print(f"Test balance:  {test_real} real / {test_fake} fake "
          f"({100 * test_real / (test_real + test_fake):.1f}% real)")
