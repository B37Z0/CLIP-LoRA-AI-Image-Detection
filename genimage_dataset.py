"""
Tiny-GenImage data pipeline for the APS360 AI-generated-image-detection
project. Loads TheKernel01/Tiny-GenImage (28k train / 7k val) from
HuggingFace and provides a cross-generator train/test split, since
generalization to unseen generators is the project's primary metric.

Install if missing:
    pip install datasets pillow transformers
"""

from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset, Subset

# Generator names per Tiny-GenImage's own `generator` class label (index 0
# is "Real"; see its dataset_info on HuggingFace).
GENERATOR_NAMES = ["Real", "ADM", "BigGAN", "GLIDE", "Midjourney", "SD14", "SD15", "VQDM", "Wukong"]


class TinyGenImageDataset(Dataset):
    def __init__(self, split: str = "train", transform: Optional[Callable] = None):
        from datasets import load_dataset

        assert split in ("train", "validation")
        self.hf_dataset = load_dataset("TheKernel01/Tiny-GenImage", split=split)
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        row = self.hf_dataset[idx]
        image = row["image"].convert("RGB")
        label = row["label"]  # 0 = real, 1 = fake
        generator = GENERATOR_NAMES[row["generator"]]

        if self.transform is not None:
            image = self.transform(image)

        return image, label, generator

    def get_generator_label(self, idx):
        """Read (label, generator) without decoding/transforming the image —
        used by CrossGeneratorSplit to build index lists cheaply."""
        row = self.hf_dataset[idx]
        return row["label"], GENERATOR_NAMES[row["generator"]]


class CrossGeneratorSplit:
    """Splits a TinyGenImageDataset so `held_out_generators` appear only in
    the test set, never in train. Real images (shared, not generator-
    specific) go into both pools."""

    def __init__(self, dataset: TinyGenImageDataset, held_out_generators: list):
        self.held_out = set(held_out_generators)

        train_idx, test_idx = [], []
        for i in range(len(dataset)):
            _, generator = dataset.get_generator_label(i)
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
    """Preprocesses a PIL image exactly as CLIP expects, using the model's
    own processor config."""
    from transformers import CLIPImageProcessor

    processor = CLIPImageProcessor.from_pretrained(model_id)

    def transform(image: Image.Image):
        return processor(images=image, return_tensors="pt")["pixel_values"][0]

    return transform


if __name__ == "__main__":
    transform = build_clip_transform()

    ds = TinyGenImageDataset(split="train", transform=transform)
    print(f"Total train samples: {len(ds)}")

    held_out = ["Midjourney", "VQDM"]
    splitter = CrossGeneratorSplit(ds, held_out_generators=held_out)
    train_set, test_set = splitter.get_splits()
    print(f"Train (excl. {held_out}): {len(train_set)} samples")
    print(f"Test (only {held_out} + real): {len(test_set)} samples")

    train_generators = {ds.get_generator_label(i)[1] for i in splitter.train_indices}
    leaked = train_generators & set(held_out)
    assert not leaked, f"held-out generators leaked into train set: {leaked}"
    print("Cross-generator split verified.")

    # Report real/fake balance per split — held-out test sets tend to be
    # dominated by real images once you exclude most generators' fakes,
    # which makes plain accuracy misleading (see printed ratio below).
    def label_counts(indices):
        real = sum(1 for i in indices if ds.get_generator_label(i)[0] == 0)
        fake = len(indices) - real
        return real, fake

    train_real, train_fake = label_counts(splitter.train_indices)
    test_real, test_fake = label_counts(splitter.test_indices)
    print(f"Train balance: {train_real} real / {train_fake} fake "
          f"({100 * train_real / (train_real + train_fake):.1f}% real)")
    print(f"Test balance:  {test_real} real / {test_fake} fake "
          f"({100 * test_real / (test_real + test_fake):.1f}% real)")
