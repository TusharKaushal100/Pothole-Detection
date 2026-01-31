import os
import shutil
import xml.etree.ElementTree as ET

# -------- PATHS --------
BASE_RDD_PATH = "RDD2022/India/train"

IMAGE_DIR = os.path.join(BASE_RDD_PATH, "images")
ANNOT_DIR = os.path.join(BASE_RDD_PATH, "annotations", "xmls")

OUTPUT_BASE = "dataset"
POTHOLE_DIR = os.path.join(OUTPUT_BASE, "pothole")
NORMAL_DIR = os.path.join(OUTPUT_BASE, "normal")

os.makedirs(POTHOLE_DIR, exist_ok=True)
os.makedirs(NORMAL_DIR, exist_ok=True)

pothole_count = 0
normal_count = 0

# -------- PROCESS XML FILES --------
for ann_file in os.listdir(ANNOT_DIR):

    if not ann_file.endswith(".xml"):
        continue

    ann_path = os.path.join(ANNOT_DIR, ann_file)
    tree = ET.parse(ann_path)
    root = tree.getroot()

    has_pothole = False

    for obj in root.iter("object"):
        label_node = obj.find("name")
        if label_node is not None:
            if label_node.text.strip() == "D40":
                has_pothole = True
                break

    filename_node = root.find("filename")
    if filename_node is None:
        continue

    image_name = filename_node.text.strip()
    src_image_path = os.path.join(IMAGE_DIR, image_name)

    if not os.path.exists(src_image_path):
        continue

    if has_pothole:
        shutil.copy(src_image_path, os.path.join(POTHOLE_DIR, image_name))
        pothole_count += 1
    else:
        shutil.copy(src_image_path, os.path.join(NORMAL_DIR, image_name))
        normal_count += 1

print("Dataset preparation complete")
print("Pothole images:", pothole_count)
print("Normal images :", normal_count)
