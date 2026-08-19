"""
Converts the ExDark (Exclusive Dark) dataset's annotation format into standard
YOLOv8 format (images/train, images/val, labels/train, labels/val, data.yaml).

ExDark ground truth files (one .txt per image) look like:
    %  <class_name> <l> <t> <w> <h> <difficult> <occlusion> <truncated>
    Bicycle 1 234 45 120 150 0 0 0

i.e. the first line is a comment/header, then one row per object with
class name + a PASCAL-VOC-style bounding box (left, top, width, height in
PIXEL coordinates), not normalized.

Download ExDark from: https://github.com/cs-chan/Exclusively-Dark-Image-Dataset
  - Images: ExDark/ (organized in per-class subfolders)
  - Annotations: ExDark_Annno/ (one .txt per image, per-class subfolders)

Usage:
    python convert_exdark_to_yolo.py \
        --images_dir /path/to/ExDark \
        --annotations_dir /path/to/ExDark_Annno \
        --output_dir ./data/exdark_yolo \
        --val_split 0.15
"""

import os
import argparse
import random
import shutil
from PIL import Image

# The 12 ExDark object classes (fixed order -> fixed class_id mapping)
EXDARK_CLASSES = [
    "Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat",
    "Chair", "Cup", "Dog", "Motorbike", "People", "Table",
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(EXDARK_CLASSES)}


def parse_exdark_annotation(txt_path, img_width, img_height):
    """Returns a list of (class_id, x_center, y_center, w, h) normalized to [0,1]."""
    boxes = []
    with open(txt_path, "r") as f:
        lines = f.readlines()

    for line in lines[1:]:  # skip header/comment line
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue

        class_name = parts[0]
        if class_name not in CLASS_TO_ID:
            continue  # skip unrecognized class names

        l, t, w, h = map(float, parts[1:5])
        class_id = CLASS_TO_ID[class_name]

        # convert pixel (left, top, width, height) -> normalized YOLO (x_center, y_center, w, h)
        x_center = (l + w / 2) / img_width
        y_center = (t + h / 2) / img_height
        w_norm = w / img_width
        h_norm = h / img_height

        # clip to valid range in case of annotation noise near image edges
        x_center = min(max(x_center, 0.0), 1.0)
        y_center = min(max(y_center, 0.0), 1.0)
        w_norm = min(max(w_norm, 0.0), 1.0)
        h_norm = min(max(h_norm, 0.0), 1.0)

        boxes.append((class_id, x_center, y_center, w_norm, h_norm))
    return boxes


def find_matching_annotation(image_path, annotations_dir):
    """ExDark annotation files are usually named '<image_filename>.txt' (keeps the image extension)."""
    base_name = os.path.basename(image_path)
    class_folder = os.path.basename(os.path.dirname(image_path))
    candidates = [
        os.path.join(annotations_dir, class_folder, base_name + ".txt"),
        os.path.join(annotations_dir, class_folder, os.path.splitext(base_name)[0] + ".txt"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def convert(images_dir, annotations_dir, output_dir, val_split=0.15, seed=42):
    random.seed(seed)

    # collect all image paths across the per-class subfolders
    image_paths = []
    for class_folder in os.listdir(images_dir):
        class_path = os.path.join(images_dir, class_folder)
        if not os.path.isdir(class_path):
            continue
        for fname in os.listdir(class_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                image_paths.append(os.path.join(class_path, fname))

    print(f"Found {len(image_paths)} images across {len(os.listdir(images_dir))} class folders")

    random.shuffle(image_paths)
    split_idx = int(len(image_paths) * (1 - val_split))
    splits = {"train": image_paths[:split_idx], "val": image_paths[split_idx:]}

    for split_name, paths in splits.items():
        img_out_dir = os.path.join(output_dir, "images", split_name)
        lbl_out_dir = os.path.join(output_dir, "labels", split_name)
        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(lbl_out_dir, exist_ok=True)

        skipped = 0
        for img_path in paths:
            ann_path = find_matching_annotation(img_path, annotations_dir)
            if ann_path is None:
                skipped += 1
                continue

            try:
                with Image.open(img_path) as im:
                    w, h = im.size
            except Exception:
                skipped += 1
                continue

            boxes = parse_exdark_annotation(ann_path, w, h)
            if not boxes:
                skipped += 1
                continue

            # copy image
            new_img_name = os.path.basename(img_path)
            shutil.copy(img_path, os.path.join(img_out_dir, new_img_name))

            # write YOLO label file
            label_name = os.path.splitext(new_img_name)[0] + ".txt"
            with open(os.path.join(lbl_out_dir, label_name), "w") as f:
                for class_id, xc, yc, bw, bh in boxes:
                    f.write(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        print(f"  {split_name}: {len(paths) - skipped} images converted, {skipped} skipped (no/empty annotation)")

    # write data.yaml
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(output_dir)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(EXDARK_CLASSES)}\n")
        f.write(f"names: {EXDARK_CLASSES}\n")

    print(f"\nDone. YOLO dataset written to {output_dir}")
    print(f"data.yaml -> {yaml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True, help="Path to ExDark/ (per-class image subfolders)")
    parser.add_argument("--annotations_dir", required=True, help="Path to ExDark_Annno/ (per-class annotation subfolders)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--val_split", type=float, default=0.15)
    args = parser.parse_args()
    convert(args.images_dir, args.annotations_dir, args.output_dir, args.val_split)
