"""
Train the ensemble classifier on cropped, enhanced, detected object images.

Expected data layout (standard ImageFolder format):
    dataset/
        train/
            class_a/*.jpg
            class_b/*.jpg
        val/
            class_a/*.jpg
            class_b/*.jpg

Usage:
    python train_ensemble.py --data_dir ./data/classification --epochs 30 --batch_size 32
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from ensemble_model import EnsembleClassifier, get_transforms


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        probs = model(imgs)                    # ensembled probabilities (already softmax'd)
        loss = criterion(torch.log(probs + 1e-8), labels)  # NLLLoss on log-probs
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = probs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        probs = model(imgs)
        loss = criterion(torch.log(probs + 1e-8), labels)

        total_loss += loss.item() * imgs.size(0)
        preds = probs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_ds = ImageFolder(os.path.join(args.data_dir, "train"), transform=get_transforms(train=True))
    val_ds = ImageFolder(os.path.join(args.data_dir, "val"), transform=get_transforms(train=False))
    num_classes = len(train_ds.classes)
    print(f"Found {num_classes} classes: {train_ds.classes}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model_names = args.models.split(",") if args.models else None
    model = EnsembleClassifier(num_classes=num_classes, model_names=model_names,
                                pretrained=not args.no_pretrained).to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": train_ds.classes,
                "model_names": model.model_names,
                "val_acc": val_acc,
            }, os.path.join(args.out_dir, "ensemble_best.pt"))
            print(f"  -> New best model saved (val_acc={val_acc:.4f})")

    print(f"Training complete. Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Root dir with train/ and val/ ImageFolder subdirs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--models", default=None,
                         help="Comma-separated subset, e.g. 'resnet50,vgg19'. Default: all 5.")
    parser.add_argument("--no_pretrained", action="store_true", help="Train from scratch (not recommended)")
    parser.add_argument("--out_dir", default="./checkpoints")
    args = parser.parse_args()
    main(args)
