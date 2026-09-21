"""
Training loop for the dual-stream CNN baseline (Yousaf et al. 2022 ref).

Uses the same per-generator / balanced accuracy evaluation as train.py since
this is used for comparison against CLIP-based results.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tinygenimage_dataset import TinyGenImageDataset, CrossGeneratorSplit
from baseline_cnn import DualStreamCNN, build_cnn_transform
from junk.train_cliploracnn import evaluate, train_one_epoch # same logic


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    held_out = ["Midjourney", "VQDM"]
    batch_size = 16
    epochs = 5
    lr = 1e-4 
    checkpoint_path = "checkpoint_cnn_baseline.pt"

    transform = build_cnn_transform()
    full_train = TinyGenImageDataset(split="train", transform=transform)
    splitter = CrossGeneratorSplit(full_train, held_out_generators=held_out)
    train_set, test_set = splitter.get_splits()

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    model = DualStreamCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_mean_bal_acc = -1.0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        eval_results = evaluate(model, test_loader, device)

        print(f"\nEpoch {epoch}/{epochs} - train loss: {train_loss:.4f}")
        for gen, acc in eval_results.items():
            print(f"  {gen}: {acc:.3f}")

        bal_accs = []
        if "Real" in eval_results:
            for gen in held_out:
                if gen in eval_results:
                    bal_acc = (eval_results["Real"] + eval_results[gen]) / 2
                    print(f" - Balanced acc (Real vs {gen}): {bal_acc:.3f}")
                    bal_accs.append(bal_acc)

        if bal_accs:
            mean_bal_acc = sum(bal_accs) / len(bal_accs)
            print(f" - Mean balanced acc: {mean_bal_acc:.3f}")
            if mean_bal_acc > best_mean_bal_acc:
                best_mean_bal_acc = mean_bal_acc
                torch.save(model.state_dict(), checkpoint_path)
                print(f" - New best - saved {checkpoint_path}")

    print(f"\nBest mean balanced acc: {best_mean_bal_acc:.3f} (checkpoint: {checkpoint_path})")


if __name__ == "__main__":
    main()
