"""
Full end-to-end inference pipeline:

    low-light image -> [Zero-DCE / Retinex / MIRNet] -> enhanced image
                     -> [YOLOv8] -> bounding boxes / crops
                     -> [5-model ensemble classifier] -> class label per crop

Usage:
    python pipeline.py --image path/to/dark_photo.jpg \
        --enhance_method zero_dce --enhance_weights checkpoints/zero_dce_final.pt \
        --yolo_weights checkpoints/yolov8_best.pt \
        --ensemble_weights checkpoints/ensemble_best.pt \
        --output_dir ./outputs
"""

import os
import argparse
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
from torchvision import transforms

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "enhancement"))
sys.path.append(os.path.join(os.path.dirname(__file__), "detection"))
sys.path.append(os.path.join(os.path.dirname(__file__), "classification"))

from zero_dce import ZeroDCE
from mirnet import MIRNet
from retinex import MSRCR
from yolov8_detector import detect_and_crop
from ensemble_model import EnsembleClassifier, get_transforms


def enhance_image(image_path, method, weights_path, device, out_path):
    """Runs the chosen enhancement method and saves the result to out_path."""
    if method == "retinex":
        img = cv2.imread(image_path)
        result = MSRCR(img)
        cv2.imwrite(out_path, result)
        return out_path

    # Learned methods (Zero-DCE / MIRNet)
    img = Image.open(image_path).convert("RGB")
    to_tensor = transforms.ToTensor()
    x = to_tensor(img).unsqueeze(0).to(device)

    if method == "zero_dce":
        model = ZeroDCE().to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        with torch.no_grad():
            enhanced, _ = model(x)
    elif method == "mirnet":
        model = MIRNet().to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        with torch.no_grad():
            enhanced = model(x)
    else:
        raise ValueError(f"Unknown enhancement method: {method}")

    enhanced_img = enhanced.squeeze(0).clamp(0, 1).cpu()
    to_pil = transforms.ToPILImage()
    to_pil(enhanced_img).save(out_path)
    return out_path


def classify_crop(crop_pil, model, classes, device):
    transform = get_transforms(train=False)
    x = transform(crop_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = model(x)
    conf, pred_idx = probs.max(dim=1)
    return classes[pred_idx.item()], conf.item()


def draw_results(image_path, detections_with_labels, out_path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for det in detections_with_labels:
        x1, y1, x2, y2 = det["bbox"]
        label = f"{det['class_label']} ({det['class_conf']:.2f})"
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, max(y1 - 12, 0)), label, fill="red")
    img.save(out_path)
    return out_path


def run_pipeline(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Enhancement
    enhanced_path = os.path.join(args.output_dir, "enhanced.jpg")
    if args.enhance_method != "none":
        print(f"[1/3] Enhancing image with {args.enhance_method}...")
        enhance_image(args.image, args.enhance_method, args.enhance_weights, device, enhanced_path)
    else:
        enhanced_path = args.image
    print(f"      -> {enhanced_path}")

    # 2. Detection
    print("[2/3] Running YOLOv8 detection...")
    crops_dir = os.path.join(args.output_dir, "crops")
    detections = detect_and_crop(args.yolo_weights, enhanced_path,
                                  conf_threshold=args.conf, save_crops_dir=crops_dir)
    print(f"      -> {len(detections)} objects detected")

    # 3. Classification
    print("[3/3] Classifying detected objects with the 5-model ensemble...")
    ckpt = torch.load(args.ensemble_weights, map_location=device)
    classes = ckpt["classes"]
    model = EnsembleClassifier(num_classes=len(classes), model_names=ckpt["model_names"],
                                pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    for det in detections:
        label, conf = classify_crop(det["crop"], model, classes, device)
        det["class_label"] = label
        det["class_conf"] = conf
        print(f"      bbox={det['bbox']}  -> {label} ({conf:.3f})")

    # Draw + save final annotated image
    result_path = draw_results(enhanced_path, detections, os.path.join(args.output_dir, "result.jpg"))
    print(f"Done. Annotated result saved to {result_path}")
    return detections


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input low-light image")
    parser.add_argument("--enhance_method", choices=["zero_dce", "mirnet", "retinex", "none"], default="zero_dce")
    parser.add_argument("--enhance_weights", default=None, help="Required for zero_dce / mirnet")
    parser.add_argument("--yolo_weights", required=True)
    parser.add_argument("--ensemble_weights", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--output_dir", default="./outputs")
    args = parser.parse_args()
    run_pipeline(args)
