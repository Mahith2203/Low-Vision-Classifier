"""
Evaluation utilities:
  - Detection: mAP@50, mAP@50-95 via YOLOv8's built-in validator
  - Classification: accuracy, precision/recall/F1, confusion matrix for the ensemble

Usage:
    python evaluate.py detection --weights checkpoints/yolov8_best.pt --data_yaml data/data.yaml
    python evaluate.py classification --weights checkpoints/ensemble_best.pt --data_dir data/classification
"""

import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "detection"))
sys.path.append(os.path.join(os.path.dirname(__file__), "classification"))

from yolov8_detector import evaluate_yolov8
from ensemble_model import EnsembleClassifier, get_transforms


def evaluate_classification(weights_path, data_dir, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(weights_path, map_location=device)
    classes = ckpt["classes"]

    val_ds = ImageFolder(os.path.join(data_dir, "val"), transform=get_transforms(train=False))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    model = EnsembleClassifier(num_classes=len(classes), model_names=ckpt["model_names"],
                                pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_preds, all_labels = [], []
    # Also track per-backbone accuracy to show ensemble benefit
    per_model_correct = {name: 0 for name in ckpt["model_names"]}
    total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            ensembled_probs, individual_logits = model(imgs, return_individual=True)

            preds = ensembled_probs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            total += imgs.size(0)

            for name, logits in individual_logits.items():
                model_preds = logits.argmax(dim=1)
                per_model_correct[name] += (model_preds == labels).sum().item()

    print("\n=== Ensemble Classification Report ===")
    print(classification_report(all_labels, all_preds, target_names=classes, digits=4))
    print("Overall accuracy:", accuracy_score(all_labels, all_preds))

    print("\n=== Per-Backbone Accuracy (for comparison) ===")
    for name, correct in per_model_correct.items():
        print(f"  {name:12s}: {correct/total:.4f}")

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="task", required=True)

    p_det = sub.add_parser("detection")
    p_det.add_argument("--weights", required=True)
    p_det.add_argument("--data_yaml", required=True)

    p_cls = sub.add_parser("classification")
    p_cls.add_argument("--weights", required=True)
    p_cls.add_argument("--data_dir", required=True)
    p_cls.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()

    if args.task == "detection":
        evaluate_yolov8(args.weights, args.data_yaml)
    else:
        evaluate_classification(args.weights, args.data_dir, args.batch_size)
