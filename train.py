"""
Training loop for the CLIP-LoRA detector.

Reports per-generator accuracy on eval instead of single number.
The test split is ~78% real, so direct accuracy would be misleading.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tinygenimage_dataset import TinyGenImageDataset, CrossGeneratorSplit, build_clip_transform
from model import CLIPLoRADetector


def evaluate(model, loader, device):
    """
    Returns {generator: accuracy} where generator is "Real" or one of the
    held out generators. Accuracy is computed per-source.
    """
    model.eval()
    correct = {}
    total = {}

    with torch.no_grad():
        for pixel_values, labels, generators in loader:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)

            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits = model(pixel_values)
            preds = logits.argmax(dim=1)

            for pred, label, gen in zip(preds.cpu(), labels.cpu(), generators):
                total[gen] = total.get(gen, 0) + 1
                if pred.item() == label.item():
                    correct[gen] = correct.get(gen, 0) + 1

    return {gen: correct.get(gen, 0) / total[gen] for gen in total}


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for pixel_values, labels, _ in loader:
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(pixel_values)
            loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    held_out = ["Midjourney", "VQDM"]
    # N=16 is comfortably below the measured ~4.85GB/batch-8.
    # Raise N for more throughput, but be careful not to blow up the GPU.
    batch_size = 16
    # LoRA on a frozen backbone converges FAST. Raise only if
    # train/eval curves are underfitting.
    epochs = 3

    transform = build_clip_transform()
    full_train = TinyGenImageDataset(split="train", transform=transform)
    splitter = CrossGeneratorSplit(full_train, held_out_generators=held_out)
    train_set, test_set = splitter.get_splits()

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    model = CLIPLoRADetector().to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        eval_results = evaluate(model, test_loader, device)

        print(f"\nEpoch {epoch}/{epochs} - train loss: {train_loss:.4f}")
        for gen, acc in eval_results.items():
            print(f" - {gen}: {acc:.3f}")

        # per-generator balanced acc = (real-recall + generator fake-recall) / 2
        # The raw group sizes (14000 real vs 2000 per generator) make a pooled number useless.
        if "Real" in eval_results:
            for gen in held_out:
                if gen in eval_results:
                    bal_acc = (eval_results["Real"] + eval_results[gen]) / 2
                    print(f"  Balanced acc (Real vs {gen}): {bal_acc:.3f}")

    torch.save(model.state_dict(), "checkpoint.pt")
    print("\nSaved checkpoint.pt")


if __name__ == "__main__":
    main()