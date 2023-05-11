import os
import shutil
import json
import numpy as np


# Set up directories
src_dir = "./OR_Intertidal_ExpPatch_ImageAnalysis-2/ExpPatch-Pics/ExpPatchPics-Processed"
annotations_file = "./coco_annotations.json"
output_base_dir = "./P2BNet/TOV_mmdetection/data/coco"


# Create output directories
os.makedirs(os.path.join(output_base_dir, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(output_base_dir, "images", "val"), exist_ok=True)
os.makedirs(os.path.join(output_base_dir, "images", "test"), exist_ok=True)
os.makedirs(os.path.join(output_base_dir, "annotations"), exist_ok=True)


# Load COCO annotations
with open(annotations_file) as f:
   data = json.load(f)


# Divide the dataset
image_ids = [img["id"] for img in data["images"]]
np.random.shuffle(image_ids)


train_ratio = 0.7
val_ratio = 0.2


train_ids = image_ids[:int(len(image_ids) * train_ratio)]
val_ids = image_ids[int(len(image_ids) * train_ratio):int(len(image_ids) * (train_ratio + val_ratio))]
test_ids = image_ids[int(len(image_ids) * (train_ratio + val_ratio)):]


# Find image file in subdirectories
def find_image(img_file_name, src_dir):
   for root, _, files in os.walk(src_dir):
       if img_file_name in files:
           return os.path.join(root, img_file_name)
   return None


def create_subset(ids, output_dir, annotation_file):
    images = [img for img in data["images"] if img["id"] in ids]
    annotations = [ann for ann in data["annotations"] if ann["image_id"] in ids]

    # Copy images
    for img in images:
        img_path = find_image(img["file_name"], src_dir)
        if img_path is not None:
            shutil.copy(img_path, os.path.join(output_base_dir, "images", output_dir))
        else:
            print(f"Image file {img['file_name']} not found. Exiting.")
            exit(1)

    # Create new COCO annotation file
    new_data = {
        "images": images,
        "annotations": annotations,
        "categories": data["categories"],
    }

    # Check if the keys exist before accessing them
    if "info" in data:
        new_data["info"] = data["info"]
    if "licenses" in data:
        new_data["licenses"] = data["licenses"]

    with open(os.path.join(output_base_dir, "annotations", annotation_file), "w") as f:
        json.dump(new_data, f, indent=2)




create_subset(train_ids, "train", "train_annotations.json")
create_subset(val_ids, "val", "val_annotations.json")
create_subset(test_ids, "test", "test_annotations.json")
