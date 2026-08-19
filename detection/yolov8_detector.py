"""
YOLOv8 object detection stage — trains/runs on ENHANCED images (output of the
low-light enhancement stage), producing bounding boxes for the classification
stage to crop and classify.

Uses the `ultralytics` package (pip install ultralytics).

Expected dataset layout (standard YOLO format):
    dataset/
        images/train/*.jpg
        images/val/*.jpg
        labels/train/*.txt      # each line: class_id x_center y_center width height (normalized 0-1)
        labels/val/*.txt
        data.yaml                # class names + paths, see make_data_yaml() below
"""

import os
import yaml
from ultralytics import YOLO


def make_data_yaml(dataset_root: str, class_names: list, out_path: str = None) -> str:
    """Generate the data.yaml file YOLOv8 needs to find images/labels and class names."""
    out_path = out_path or os.path.join(dataset_root, "data.yaml")
    data = {
        "path": os.path.abspath(dataset_root),
        "train": "images/train",
        "val": "images/val",
        "nc": len(class_names),
        "names": class_names,
    }
    with open(out_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)
    return out_path


def train_yolov8(data_yaml: str, epochs: int = 100, imgsz: int = 640,
                  batch: int = 16, model_size: str = "yolov8n.pt", out_dir: str = "./runs/detect"):
    """
    model_size options: yolov8n.pt (nano, fastest) / yolov8s.pt / yolov8m.pt /
                         yolov8l.pt / yolov8x.pt (most accurate, slowest)
    """
    model = YOLO(model_size)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=out_dir,
        name="low_vision_yolov8",
        patience=20,          # early stopping
        augment=True,
        mosaic=1.0,
        mixup=0.1,
    )
    return model, results


def evaluate_yolov8(weights_path: str, data_yaml: str):
    model = YOLO(weights_path)
    metrics = model.val(data=data_yaml)
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP75:    {metrics.box.map75:.4f}")
    return metrics


def detect_and_crop(weights_path: str, image_path: str, conf_threshold: float = 0.25,
                     save_crops_dir: str = None):
    """
    Runs detection on a single (already enhanced) image and returns a list of
    crops ready to be fed into the ensemble classifier, along with boxes/scores.
    """
    from PIL import Image

    model = YOLO(weights_path)
    results = model.predict(source=image_path, conf=conf_threshold, verbose=False)

    img = Image.open(image_path).convert("RGB")
    detections = []

    for r in results:
        boxes = r.boxes
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i].cpu().numpy())
            cls_id = int(boxes.cls[i].cpu().numpy())
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            crop = img.crop((x1, y1, x2, y2))

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": conf,
                "class_id": cls_id,
                "crop": crop,
            })

    if save_crops_dir:
        os.makedirs(save_crops_dir, exist_ok=True)
        for idx, det in enumerate(detections):
            det["crop"].save(os.path.join(save_crops_dir, f"crop_{idx}_cls{det['class_id']}.jpg"))

    return detections


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--data_yaml", required=True)
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.add_argument("--imgsz", type=int, default=640)
    p_train.add_argument("--batch", type=int, default=16)
    p_train.add_argument("--model_size", default="yolov8n.pt")

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--weights", required=True)
    p_eval.add_argument("--data_yaml", required=True)

    p_detect = sub.add_parser("detect")
    p_detect.add_argument("--weights", required=True)
    p_detect.add_argument("--image", required=True)
    p_detect.add_argument("--conf", type=float, default=0.25)
    p_detect.add_argument("--save_crops_dir", default=None)

    args = parser.parse_args()

    if args.command == "train":
        train_yolov8(args.data_yaml, args.epochs, args.imgsz, args.batch, args.model_size)
    elif args.command == "eval":
        evaluate_yolov8(args.weights, args.data_yaml)
    elif args.command == "detect":
        dets = detect_and_crop(args.weights, args.image, args.conf, args.save_crops_dir)
        for d in dets:
            print(d["bbox"], d["class_id"], f"{d['confidence']:.3f}")
