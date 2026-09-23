"""
Training loop for the CLIP-LoRA AI-image detector.

Reports per-generator accuracy on eval instead of single number.
The Tiny-GenImage test split is ~78% real, so direct accuracy would be misleading.
"""

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tinygenimage_dataset import TinyGenImageDataset, CrossGeneratorSplit, build_clip_transform
from model import CLIPLoRADetector, sample_calibration_batch


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
    has_grad_projection = hasattr(model, "project_lora_null_space_gradient")

    for pixel_values, labels, _ in loader:
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(pixel_values)
            loss = criterion(logits, labels)
        loss.backward()

        if has_grad_projection:
            model.project_lora_null_space_gradient()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    held_out = ["Midjourney", "VQDM"]
    # N=16 comfortably below the measured ~4.85GB/batch-8.
    # Raise N for more throughput but be careful.
    batch_size = 16
    # LoRA on a frozen backbone converges FAST. 
    # Raise only if curves show underfitting.
    epochs = 7
    use_lora = True # False for frozen linear-probe baseline
    use_freq_branch = False # fuse the YCbCr DFT+DWT frequency branch into the classifier
    null_space_init = True # LoRA-Null: activation-based null-space init of B, A
    null_space_grad_protect = False # LoRA-Null ablation (null_space_init=True): project A gradient during training
    checkpoint_path = (
        f"checkpoint_lora{'_null' if null_space_init else ''}"
        f"{'_gradprotect' if null_space_grad_protect else ''}"
        f"{'_freq' if use_freq_branch else ''}.pt"
        if use_lora else "checkpoint_frozen.pt"
    )
    checkpoint_dir = (
        f"checkpoints_lora{'_null' if null_space_init else ''}"
        f"{'_gradprotect' if null_space_grad_protect else ''}"
        f"{'_freq' if use_freq_branch else ''}"
        if use_lora else "checkpoints_frozen"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    resume = False # True to resume from previous checkpoint

    transform = build_clip_transform()
    full_train = TinyGenImageDataset(split="train", transform=transform)
    splitter = CrossGeneratorSplit(full_train, held_out_generators=held_out)
    train_set, test_set = splitter.get_splits()

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    calibration = None
    if null_space_init or null_space_grad_protect:
        calibration = sample_calibration_batch(full_train, n=64, device=device)

    model = CLIPLoRADetector(
        use_lora=use_lora, use_freq_branch=use_freq_branch,
        null_space_init=null_space_init, null_space_grad_protect=null_space_grad_protect,
        calibration_pixel_values=calibration,
    ).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 1 # 1 if not resuming
    best_mean_bal_acc = -1.0

    if resume and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_mean_bal_acc = ckpt["best_mean_bal_acc"]
        print(f"Resumed from {checkpoint_path} (epoch {ckpt['epoch']}, "
              f"best bal acc {best_mean_bal_acc:.3f})")

    for epoch in range(start_epoch, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        eval_results = evaluate(model, test_loader, device)
 
        print(f"\nEpoch {epoch}/{epochs} — train loss: {train_loss:.4f}")
        for gen, acc in eval_results.items():
            print(f"  {gen}: {acc:.3f}")
 
        # per-generator balanced acc = (real-recall + generator fake-recall) / 2
        # The raw group sizes (14000 real vs 2000 per generator) make a pooled number useless.
        
        # UPDATE: use this number to select the checkpoint. Train loss and even the raw
        # held-out accuracy improve/decline in frankly unfathomable ways that don't seem 
        # to track actual generalization (Midjourney drift in prev results)
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
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_mean_bal_acc": best_mean_bal_acc,
                }, checkpoint_path)
                print(f" - New best - saved {checkpoint_path}")  

        # Checkpoint every epoch regardless, can't be too safe
        epoch_ckpt = os.path.join(checkpoint_dir, f"epoch_{epoch}.pt")
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "mean_bal_acc": mean_bal_acc if bal_accs else None,
            "eval_results": eval_results,
        }, epoch_ckpt)
        print(f" - Saved {epoch_ckpt}")
 
    print(f"\nBest mean balanced acc: {best_mean_bal_acc:.3f} (checkpoint: {checkpoint_path})")
 
 
if __name__ == "__main__":
    main()
