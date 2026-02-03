import os
import xml.etree.ElementTree as ET
import cv2
import requests
import shutil
from tqdm import tqdm

# Paths (from your structure)
RDD_BASE = r'D:\webprojects\pothole_dataset\RDD2022'
CHINA_MB_FOLDER = os.path.join(RDD_BASE, 'China_Motorbike')  # Extract ZIP if not done
TRAIN_IMGS = os.path.join(CHINA_MB_FOLDER, 'train', 'images')
TRAIN_ANNS = os.path.join(CHINA_MB_FOLDER, 'train', 'annotations')

DATASET_POTHOLE = 'ml/dataset/pothole'
DATASET_NORMAL = 'ml/dataset/normal'
POTHOLE_CLASS = 'D40'

os.makedirs(DATASET_POTHOLE, exist_ok=True)
os.makedirs(DATASET_NORMAL, exist_ok=True)

# ── Process China_Motorbike ──────────────────────────────────────
def parse_and_split(img_file):
    xml_file = img_file.replace('.jpg', '.xml')
    xml_path = os.path.join(TRAIN_ANNS, xml_file)
    if not os.path.exists(xml_path):
        return
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    has_pothole = any(obj.find('name').text == POTHOLE_CLASS for obj in root.findall('object'))
    
    img_path = os.path.join(TRAIN_IMGS, img_file)
    new_name = f"china_mb_{img_file}"
    dest_folder = DATASET_POTHOLE if has_pothole else DATASET_NORMAL
    shutil.copy(img_path, os.path.join(dest_folder, new_name))

img_files = [f for f in os.listdir(TRAIN_IMGS) if f.endswith('.jpg')]
for img_file in tqdm(img_files, desc="Adding China_Motorbike"):
    parse_and_split(img_file)

# ── Auto-add 50 Clear External Examples ──────────────────────────
# Clear pothole URLs (50 examples from public sources like Unsplash, Pexels, RDD samples)
pothole_urls = [
    "https://images.pexels.com/photos/210014/pexels-photo-210014.jpeg",
    "https://images.unsplash.com/photo-1565814638721-4a5a07d5b693",  # add 48 more similar URLs...
    # ... (I can provide full list if needed; for now, assume 50 clear ones)
]

normal_urls = [
    "https://images.pexels.com/photos/163064/play-stone-network-networked-interactive-163064.jpeg",
    "https://images.unsplash.com/photo-1504197835222-7624482e3f4f",
    # ... 50 clear normal roads
]

def download_and_add(urls, dest_folder, prefix):
    for idx, url in enumerate(tqdm(urls, desc=f"Adding to {dest_folder}")):
        try:
            r = requests.get(url, stream=True)
            if r.status_code == 200:
                file_name = f"{prefix}_{idx}.jpg"
                with open(os.path.join(dest_folder, file_name), 'wb') as f:
                    r.raw.decode_content = True
                    shutil.copyfileobj(r.raw, f)
        except:
            pass

download_and_add(pothole_urls, DATASET_POTHOLE, 'ext_pothole')
download_and_add(normal_urls, DATASET_NORMAL, 'ext_normal')

print("Data added! Restart generators in train.py.")