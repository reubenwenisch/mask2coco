import json
import numpy as np
from pycocotools import mask
from skimage import measure
import os
import cv2

annotation_id = 0
black = [0,0,0]
def create_annotation_format(masks, category_id, image_id):
    global annotation_id
    annotation = {
            "segmentation": [],
            "area": [],
            "iscrowd": int(0),
            "image_id": int(image_id),
            "bbox": [],
            "category_id": int(category_id),
            "id": int(annotation_id)
        }
    ground_truth_binary_mask= cv2.copyMakeBorder(masks,1,1,1,1,cv2.BORDER_CONSTANT,value=black)
    fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask)
    encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
    ground_truth_area = mask.area(encoded_ground_truth)
    ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
    contours = measure.find_contours(ground_truth_binary_mask, 0.5)
    annotation["area"] = int(ground_truth_area)
    annotation["category_id"] = int(category_id)
    annotation["bbox"] = ground_truth_bounding_box.tolist()
    for contour in contours:
        contour = np.flip(contour, axis=1).astype(int)
        segmentation = contour.ravel().tolist()
        annotation["segmentation"].append(segmentation)
    annotation_id += 1
    return annotation


def create_category_annotation(category_dict):
    category_list = []

    for key, value in category_dict.items():
        category = {
            "supercategory": key,
            "id": int(value),
            "name": key
        }
        category_list.append(category)

    return category_list

def create_image_annotation(file_name, width, height, image_id):
    images = {
        "file_name": file_name + '.jpg',
        "height": int(height),
        "width": int(width),
        "id": int(image_id)
    }
    return images

def get_coco_json_format():
    # Standard COCO format 
    coco_format = {
        "info": {},
        "licenses": [],
        "images": [{}],
        "categories": [{}],
        "annotations": [{}]
    }

    return coco_format