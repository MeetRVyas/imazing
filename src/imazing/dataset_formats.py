"""Import/export codecs for common dataset formats: COCO, YOLO, Pascal VOC, image/mask folder pairs, and classification (ImageFolder-style) folders.

- COCO segmentation: polygon format only, not RLE-encoded masks.
- YOLO and Pascal VOC are bounding-box-only formats natively, so
    exporting a Dataset to either only carries `boxes` -- polygons/masks/
    keypoints have nowhere to go in those formats.
- Image/mask folders support one mask per image (semantic segmentation), not one mask per instance.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence
from xml.dom import minidom

import numpy as np

from .core import Imazing
from .annotations import AnnotationSet, BoundingBox, Polygon, Mask
from ._validation import as_gray


# --- COCO --------------------------------------------------------------------

def load_coco(images_dir: str | Path, annotation_file: str | Path):
    from .dataset import Dataset

    images_dir = Path(images_dir)
    with open(annotation_file, "r") as f:
        coco = json.load(f)

    categories = {c["id"]: c["name"] for c in coco.get("categories", [])}
    images_by_id = {img["id"]: img for img in coco.get("images", [])}
    annotations_by_image: dict[int, list[dict]] = {}
    for ann in coco.get("annotations", []):
        annotations_by_image.setdefault(ann["image_id"], []).append(ann)

    items = []
    for image_id, image_info in images_by_id.items():
        path = images_dir / image_info["file_name"]
        aset = AnnotationSet()
        for ann in annotations_by_image.get(image_id, []):
            label = categories.get(ann.get("category_id"))
            bbox = ann.get("bbox")
            if bbox:
                x, y, w, h = bbox
                aset.add(BoundingBox(x, y, w, h, label=label))
            seg = ann.get("segmentation")
            if seg and isinstance(seg, list) and seg and isinstance(seg[0], list):
                # Polygon segmentation only; COCO's RLE-encoded "iscrowd"
                # masks aren't decoded here (see module docstring).
                for polygon_coords in seg:
                    points = list(zip(polygon_coords[0::2], polygon_coords[1::2]))
                    aset.add(Polygon(points=points, label=label))
        items.append((str(path), aset))

    return Dataset(items)


def save_coco(dataset, output_dir: str | Path, annotation_file: str = "annotations.json", copy_images: bool = True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    categories_seen: dict[str, int] = {}
    images_json = []
    annotations_json = []
    ann_id = 1

    for i, item in enumerate(dataset):
        img = item.image
        h, w = img.image.shape[:2]
        file_name = f"{i:06d}.jpg"
        image_id = i + 1
        images_json.append({"id": image_id, "file_name": file_name, "width": w, "height": h})

        if copy_images:
            img.save(str(output_dir / file_name))

        for box in item.annotations.boxes:
            label = box.label if box.label is not None else "unlabeled"
            cat_id = categories_seen.setdefault(label, len(categories_seen) + 1)
            annotations_json.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": [box.x, box.y, box.width, box.height],
                "area": box.area,
                "iscrowd": 0,
            })
            ann_id += 1

        for poly in item.annotations.polygons:
            label = poly.label if poly.label is not None else "unlabeled"
            cat_id = categories_seen.setdefault(label, len(categories_seen) + 1)
            flat = [coord for point in poly.points for coord in point]
            box = poly.bounding_box()
            annotations_json.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": cat_id,
                "segmentation": [flat],
                "bbox": [box.x, box.y, box.width, box.height],
                "area": box.area,
                "iscrowd": 0,
            })
            ann_id += 1

    categories_json = [{"id": cid, "name": name} for name, cid in categories_seen.items()]
    coco = {"images": images_json, "annotations": annotations_json, "categories": categories_json}

    output_path = output_dir / annotation_file
    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)
    return output_path


# --- YOLO ----------------------------------------------------------------------

def load_yolo(images_dir: str | Path, labels_dir: str | Path, classes_file: str | Path):
    from .dataset import Dataset

    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    with open(classes_file) as f:
        class_names = [line.strip() for line in f if line.strip()]

    items = []
    for image_path in sorted(p for p in images_dir.iterdir() if p.is_file()):
        label_path = labels_dir / (image_path.stem + ".txt")
        aset = AnnotationSet()
        if label_path.exists():
            # YOLO stores normalized [0,1] coordinates; de-normalizing to
            # this library's pixel-space convention requires the image's
            # actual dimensions, so (unlike from_directory/from_paths)
            # each image is decoded once here, at import time.
            img_h, img_w = Imazing(str(image_path)).image.shape[:2]
            for line in label_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                cls_idx = int(parts[0])
                cx, cy, nw, nh = (float(v) for v in parts[1:5])
                label = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)
                box_w, box_h = nw * img_w, nh * img_h
                box_x, box_y = cx * img_w - box_w / 2, cy * img_h - box_h / 2
                aset.add(BoundingBox(box_x, box_y, box_w, box_h, label=label))
        items.append((str(image_path), aset))

    return Dataset(items)


def save_yolo(dataset, output_dir: str | Path, class_names: Sequence[str], copy_images: bool = True):
    output_dir = Path(output_dir)
    images_out = output_dir / "images"
    labels_out = output_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    name_to_index = {name: i for i, name in enumerate(class_names)}

    for i, item in enumerate(dataset):
        img = item.image
        h, w = img.image.shape[:2]
        file_name = f"{i:06d}.jpg"
        if copy_images:
            img.save(str(images_out / file_name))

        lines = []
        for box in item.annotations.boxes:
            if box.label not in name_to_index:
                continue  # label not in the given class list -- skip it
            cls_idx = name_to_index[box.label]
            cx = (box.x + box.width / 2) / w
            cy = (box.y + box.height / 2) / h
            lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {box.width / w:.6f} {box.height / h:.6f}")

        (labels_out / f"{i:06d}.txt").write_text("\n".join(lines))

    (output_dir / "classes.txt").write_text("\n".join(class_names))
    return output_dir


# --- Pascal VOC ------------------------------------------------------------------

def load_voc(images_dir: str | Path, annotations_dir: str | Path):
    from .dataset import Dataset

    images_dir = Path(images_dir)
    annotations_dir = Path(annotations_dir)

    items = []
    for xml_path in sorted(annotations_dir.glob("*.xml")):
        root = ET.parse(xml_path).getroot()
        filename = root.findtext("filename") or (xml_path.stem + ".jpg")
        image_path = images_dir / filename

        aset = AnnotationSet()
        for obj in root.findall("object"):
            name = obj.findtext("name")
            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue
            xmin = float(bndbox.findtext("xmin"))
            ymin = float(bndbox.findtext("ymin"))
            xmax = float(bndbox.findtext("xmax"))
            ymax = float(bndbox.findtext("ymax"))
            aset.add(BoundingBox(xmin, ymin, xmax - xmin, ymax - ymin, label=name))

        items.append((str(image_path), aset))

    return Dataset(items)


def save_voc(dataset, output_dir: str | Path, copy_images: bool = True):
    output_dir = Path(output_dir)
    images_out = output_dir / "JPEGImages"
    annotations_out = output_dir / "Annotations"
    images_out.mkdir(parents=True, exist_ok=True)
    annotations_out.mkdir(parents=True, exist_ok=True)

    for i, item in enumerate(dataset):
        img = item.image
        h, w = img.image.shape[:2]
        c = img.channels
        file_name = f"{i:06d}.jpg"
        if copy_images:
            img.save(str(images_out / file_name))

        root = ET.Element("annotation")
        ET.SubElement(root, "filename").text = file_name
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(w)
        ET.SubElement(size, "height").text = str(h)
        ET.SubElement(size, "depth").text = str(c)

        for box in item.annotations.boxes:
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = box.label or "unlabeled"
            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(int(round(box.x)))
            ET.SubElement(bndbox, "ymin").text = str(int(round(box.y)))
            ET.SubElement(bndbox, "xmax").text = str(int(round(box.x2)))
            ET.SubElement(bndbox, "ymax").text = str(int(round(box.y2)))

        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
        (annotations_out / f"{i:06d}.xml").write_text(xml_str)

    return output_dir


# --- Image / mask folders ---------------------------------------------------------

def load_image_mask_folders(images_dir: str | Path, masks_dir: str | Path):
    from .dataset import Dataset

    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)

    items = []
    for image_path in sorted(p for p in images_dir.iterdir() if p.is_file()):
        mask_candidates = sorted(masks_dir.glob(image_path.stem + ".*"))
        aset = AnnotationSet()
        if mask_candidates:
            mask_img = Imazing(str(mask_candidates[0]))
            mask_data = mask_img.image
            if mask_data.ndim == 3:
                mask_data = as_gray(mask_data, mask_img.color_space)
            aset.add(Mask(data=mask_data))
        items.append((str(image_path), aset))

    return Dataset(items)


def save_image_mask_folders(dataset, output_dir: str | Path):
    output_dir = Path(output_dir)
    images_out = output_dir / "images"
    masks_out = output_dir / "masks"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    for i, item in enumerate(dataset):
        file_name = f"{i:06d}.png"
        item.image.save(str(images_out / file_name))
        if item.annotations.masks:
            # Only the first mask is exported -- this format is for
            # semantic segmentation (one mask per image), not one mask
            # per instance.
            mask = item.annotations.masks[0]
            Imazing(mask.data.astype(np.uint8)).save(str(masks_out / file_name))

    return output_dir


# --- Classification (ImageFolder-style) folders -----------------------------------

def save_class_folders(dataset, output_dir: str | Path):
    output_dir = Path(output_dir)
    counters: dict[str, int] = {}

    for item in dataset:
        class_name = item.annotations.labels[0].name if item.annotations.labels else "unlabeled"
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        idx = counters.get(class_name, 0)
        counters[class_name] = idx + 1
        item.image.save(str(class_dir / f"{idx:06d}.jpg"))

    return output_dir