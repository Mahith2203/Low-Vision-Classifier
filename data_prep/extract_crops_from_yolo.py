"""
Auto-generates the classification-stage dataset by cropping objects out of
images using existing YOLO-format detection labels (e.g. the output of
convert_exdark_to_yolo.py, or your own detection annotations).

This turns:
    images/train/img001.jpg
    labels/train/img001.txt   (class_id x_center y_center w h, normalized)
into:
    classification_data/train/<class_name>/img001_obj0.jpg
    classification_data/train/<class_name>/img001_obj1.jpg
    ...
i.e. the exact ImageFolder layout train_ensemble.py expects.

Usage:
    python extract_crops_from_yolo.py \
        --yolo_dir ./data/exdark_yolo \
        --output_dir ./data/classification \
        --class_names Bicycle Boat Bottle Bus Car Cat Chair Cup Dog Motorbike People Table \
        --padding 0.1
"""

import os
import argparse
import glob
from PIL import Image


def yolo_box_to_pixels(x_center, y_center, w, h, img_w, img_h, padding=0.0):
    """Converts normalized YOLO box to pixel (x1,y1,x2,y2), with optional padding fraction."""
    box_w = w * img_w
    box_h = h * img_h
    cx = x_center * img_w
    cy = y_center * img_h

    # add padding (helps classifier see a bit of context around the object)
    box_w *= (1 + padding)
    box_h *= (1 + padding)

    x1 = max(0, int(cx - box_w / 2))
    y1 = max(0, int(cy - box_h / 2))
    x2 = min(img_w, int(cx + box_w / 2))
    y2 = min(img_h, int(cy + box_h / 2))
    return x1, y1, x2, y2


def extract_crops(yolo_dir, output_dir, class_names, padding=0.1, min_size=20):
    for split in ["train", "val"]:
        images_dir = os.path.join(yolo_dir, "images", split)
        labels_dir = os.path.join(yolo_dir, "labels", split)

        if not os.path.isdir(images_dir):
            print(f"Skipping split '{split}' (no {images_dir})")
            continue

        label_paths = sorted(glob.glob(os.path.join(labels_dir, "*.txt")))
        print(f"[{split}] Found {len(label_paths)} label files")

        crop_count = 0
        skipped_small = 0

        for label_path in label_paths:
            base_name = os.path.splitext(os.path.basename(label_path))[0]

            # find matching image (try common extensions)
            img_path = None
            for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                candidate = os.path.join(images_dir, base_name + ext)
                if os.path.exists(candidate):
                    img_path = candidate
                    break
            if img_path is None:
                continue

            img = Image.open(img_path).convert("RGB")
            img_w, img_h = img.size

            with open(label_path, "r") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]

            for obj_idx, line in enumerate(lines):
                parts = line.split()
                if len(parts) < 5:
                    continue
                class_id = int(parts[0])
                x_center, y_center, w, h = map(float, parts[1:5])

                if class_id >= len(class_names):
                    continue
                class_name = class_names[class_id]

                x1, y1, x2, y2 = yolo_box_to_pixels(x_center, y_center, w, h, img_w, img_h, padding)

                if (x2 - x1) < min_size or (y2 - y1) < min_size:
                    skipped_small += 1
                    continue

                crop = img.crop((x1, y1, x2, y2))

                out_class_dir = os.path.join(output_dir, split, class_name)
                os.makedirs(out_class_dir, exist_ok=True)
                crop.save(os.path.join(out_class_dir, f"{base_name}_obj{obj_idx}.jpg"))
                crop_count += 1

        print(f"[{split}] Extracted {crop_count} crops ({skipped_small} skipped: too small)")

    print(f"\nDone. Classification dataset written to {output_dir}")
    print("Layout matches train_ensemble.py's expected --data_dir structure (train/<class>, val/<class>)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo_dir", required=True, help="YOLO dataset root (has images/ and labels/)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--class_names", nargs="+", required=True,
                         help="Class names in class_id order, e.g. matching data.yaml's 'names' list")
    parser.add_argument("--padding", type=float, default=0.1, help="Fractional padding added around each box")
    parser.add_argument("--min_size", type=int, default=20, help="Skip crops smaller than this (pixels)")
    args = parser.parse_args()
    extract_crops(args.yolo_dir, args.output_dir, args.class_names, args.padding, args.min_size)
