import json
import os
from pathlib import Path

LABEL_NAME = "wally"
CLASS_ID = 0

def convert(json_path):
    with open(json_path) as f:
        data = json.load(f)

    img_w = data["imageWidth"]
    img_h = data["imageHeight"]

    yolo_lines = []

    for shape in data["shapes"]:
        if shape["label"] != LABEL_NAME:
            continue

        (x1, y1), (x2, y2) = shape["points"]

        x_center = ((x1 + x2) / 2) / img_w
        y_center = ((y1 + y2) / 2) / img_h
        width = abs(x2 - x1) / img_w
        height = abs(y2 - y1) / img_h

        yolo_lines.append(
            f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

    return yolo_lines


def main():
    json_dir = "images"   # donde están los .json
    label_dir = "labels"
    os.makedirs(label_dir, exist_ok=True)

    for json_file in Path(json_dir).glob("*.json"):
        yolo_lines = convert(json_file)
        if not yolo_lines:
            continue

        txt_path = Path(label_dir) / (json_file.stem + ".txt")
        with open(txt_path, "w") as f:
            f.write("\n".join(yolo_lines))


if __name__ == "__main__":
    main()
