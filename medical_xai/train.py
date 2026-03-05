"""
Training script for Pneumonia classification (ResNet18 FusionModel).

Dataset: Chest X-Ray Images (Pneumonia)
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Expected folder structure after unzipping:
    data/chest_xray/
        train/
            NORMAL/   *.jpeg
            PNEUMONIA/ *.jpeg
        val/
            NORMAL/
            PNEUMONIA/
        test/
            NORMAL/
            PNEUMONIA/

Usage:
    python train.py
    python train.py --epochs 20 --lr 0.0001 --batch_size 32
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from models.fusion_model import FusionModel
from utils.preprocessing import get_transform


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train FusionModel on Chest X-Ray dataset")
    p.add_argument("--data_dir",   type=str, default="data/chest_xray",   help="Path to chest_xray folder")
    p.add_argument("--weights_dir",type=str, default="weights",           help="Where to save model weights")
    p.add_argument("--epochs",     type=int, default=15,                  help="Training epochs")
    p.add_argument("--lr",         type=float, default=3e-4,              help="Learning rate")
    p.add_argument("--batch_size", type=int, default=32,                  help="Batch size")
    p.add_argument("--freeze_backbone", action="store_true",              help="Freeze CNN backbone (transfer learning)")
    p.add_argument("--image_size", type=int, default=224)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)

        if (batch_idx + 1) % 20 == 0:
            print(f"  Epoch {epoch} [{batch_idx+1}/{len(loader)}] "
                  f"Loss: {loss.item():.4f}  Acc: {correct/total:.3f}")

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss   = criterion(logits, labels)
            preds  = logits.argmax(1)

            total_loss += loss.item() * imgs.size(0)
            correct    += (preds == labels).sum().item()
            total      += imgs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def save_training_curves(train_losses, val_losses, train_accs, val_accs, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(train_losses) + 1)

    axes[0].plot(epochs, train_losses, "b-o", label="Train Loss")
    axes[0].plot(epochs, val_losses,   "r-o", label="Val Loss")
    axes[0].set_title("Loss Curves"); axes[0].legend(); axes[0].grid(True)

    axes[1].plot(epochs, train_accs, "b-o", label="Train Acc")
    axes[1].plot(epochs, val_accs,   "r-o", label="Val Acc")
    axes[1].set_title("Accuracy Curves"); axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close()
    print(f"  📊 Training curves saved to {save_dir}/training_curves.png")


def save_confusion_matrix(labels, preds, class_names, save_dir):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=150)
    plt.close()
    print(f"  📊 Confusion matrix saved to {save_dir}/confusion_matrix.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    # ── Paths ──────────────────────────────────────────────────────────────
    data_dir    = Path(args.data_dir)
    weights_dir = Path(args.weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)

    if not (data_dir / "train").exists():
        print(f"\n❌ Dataset not found at '{data_dir}'.")
        print("Please download from: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
        print("Unzip to: data/chest_xray/")
        return

    # ── Datasets ────────────────────────────────────────────────────────────
    train_transform = get_transform(args.image_size, augment=True)
    val_transform   = get_transform(args.image_size, augment=False)

    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_transform)
    val_ds   = datasets.ImageFolder(data_dir / "val",   transform=val_transform)
    test_ds  = datasets.ImageFolder(data_dir / "test",  transform=val_transform)

    class_names = train_ds.classes  # ['NORMAL', 'PNEUMONIA']
    print(f"📂 Classes: {class_names}")
    print(f"   Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    # Class weights for imbalance (Pneumonia > Normal)
    labels_all = [s[1] for s in train_ds.samples]
    class_counts = np.bincount(labels_all)
    class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # ── Model ───────────────────────────────────────────────────────────────
    model = FusionModel(num_classes=2).to(device)

    if args.freeze_backbone:
        for param in model.cnn.parameters():
            param.requires_grad = False
        print("🔒 CNN backbone frozen (training classifier head only)")
        optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training loop ────────────────────────────────────────────────────────
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    print(f"\n🚀 Starting training for {args.epochs} epochs...\n")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(tr_loss); val_losses.append(va_loss)
        train_accs.append(tr_acc);   val_accs.append(va_acc)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
              f"Val Loss: {va_loss:.4f} Acc: {va_acc:.4f} | "
              f"Time: {elapsed:.1f}s")

        # Save best model
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            save_path = weights_dir / "resnet18_pneumonia_classifier.pth"
            torch.save(model.state_dict(), str(save_path))
            print(f"  ✅ Best model saved (val_acc={best_val_acc:.4f})")

    # ── Test evaluation ──────────────────────────────────────────────────────
    print("\n📊 Evaluating on test set...")
    model.load_state_dict(torch.load(str(weights_dir / "resnet18_pneumonia_classifier.pth"), map_location=device))
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)

    print(f"\nTest Loss: {test_loss:.4f}  Test Acc: {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=class_names))

    save_training_curves(train_losses, val_losses, train_accs, val_accs, str(weights_dir))
    save_confusion_matrix(test_labels, test_preds, class_names, str(weights_dir))

    print(f"\n🎉 Training complete! Best val accuracy: {best_val_acc:.4f}")
    print(f"   Weights saved to: {weights_dir / 'resnet18_pneumonia_classifier.pth'}")


if __name__ == "__main__":
    main()
