#!/usr/bin/python

# pip install lxml

import sys
import os
import json
import xml.etree.ElementTree as ET
from pycocotools.coco import COCO
from pycocotools import mask
import glob
import numpy as np
from skimage import measure
from PIL import Image

START_BOUNDING_BOX_ID = 1
PRE_DEFINE_CATEGORIES = None
# If necessary, pre-define category and its id
#  PRE_DEFINE_CATEGORIES = {"aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4,
#  "bottle":5, "bus": 6, "car": 7, "cat": 8, "chair": 9,
#  "cow": 10, "diningtable": 11, "dog": 12, "horse": 13,
#  "motorbike": 14, "person": 15, "pottedplant": 16,
#  "sheep": 17, "sofa": 18, "train": 19, "tvmonitor": 20}


def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise ValueError("Can not find %s in %s." % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise ValueError(
            "The size of %s is supposed to be %d, but is %d."
            % (name, length, len(vars))
        )
    if length == 1:
        vars = vars[0]
    return vars

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation 
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons


def get_filename_as_int(filename):
    try:
        filename = filename.replace("\\", "/")
        filename = os.path.splitext(os.path.basename(filename))[0]
        return int(filename)
    except:
        raise ValueError("Filename %s is supposed to be an integer." % (filename))


def get_categories(xml_files):
    """Generate category name to id mapping from a list of xml files.
    
    Arguments:
        xml_files {list} -- A list of xml file paths.
    
    Returns:
        dict -- category name to id mapping.
    """
    classes_names = []
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall("object"):
            classes_names.append(member[0].text)
    classes_names = list(set(classes_names))
    classes_names.sort()
    return {name: (i+1) for i, name in enumerate(classes_names)}


def convert(xml_files, json_file, mask_dir, train_files, val=False, inst_dir=None):
    json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}
    if PRE_DEFINE_CATEGORIES is not None:
        categories = PRE_DEFINE_CATEGORIES
    else:
        categories = get_categories(xml_files)
    bnd_id = START_BOUNDING_BOX_ID
    now=0
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        path = get(root, "path")
        if len(path) == 1:
            filename = os.path.basename(path[0].text)
        elif len(path) == 0:
            filename = get_and_check(root, "filename", 1).text
        else:
            raise ValueError("%d paths found in %s" % (len(path), xml_file))
        mask_path = os.path.join(mask_dir, filename[:-4] + '.png')
        if not (filename[:-4] in train_files):
            print ('skip this %s'%filename)
            continue
        now+=1
        #The filename must be a number
        image_id = get_filename_as_int(filename)
        size = get_and_check(root, "size", 1)
        width = int(get_and_check(size, "width", 1).text)
        height = int(get_and_check(size, "height", 1).text)
        image = {
            "file_name": filename,
            "height": height,
            "width": width,
            "id": image_id,
        }
        json_dict["images"].append(image)

        mask_cls = np.asarray(Image.open(mask_path), dtype=np.int32)
        if val:
            inst_path = os.path.join(inst_dir, filename[:-4] + '.png')
            inst_mask = np.asarray(Image.open(inst_path), dtype=np.int32)
            this_mask = inst_mask.copy()
            inst_id_list = np.unique(inst_mask)
            sum_pix = 0
            max_id = 0
            for inst in inst_id_list:
                if inst==0 or inst==255:
                    continue
                this_mask = this_mask*0.0
                this_mask[inst_mask==inst]=1
                category_id = np.unique(mask_cls[this_mask==1])[0]
                this_mask = np.array(this_mask).astype(np.uint8)
                segmentation = binary_mask_to_polygon(this_mask, tolerance=2)
                binary_mask_encoded = mask.encode(np.asfortranarray(this_mask.astype(np.uint8)))
                area = mask.area(binary_mask_encoded)
                bounding_box = mask.toBbox(binary_mask_encoded)
                if segmentation ==[]:
                    this_mask = inst_mask.copy()
                    this_mask = this_mask*0.0
                    xmin = int(bounding_box[0])
                    xmax = int(bounding_box[0]+bounding_box[2])
                    ymin = int(bounding_box[1])
                    ymax = int(bounding_box[1]+bounding_box[3])
                    this_mask[ymin:ymax,xmin:xmax]=1
                    this_mask = np.array(this_mask).astype(np.uint8)
                    segmentation = binary_mask_to_polygon(this_mask, tolerance=2)
                if segmentation==[]:
                    continue
                ann = {
                    "area": area.tolist(),#o_width * o_height,
                    "iscrowd": 0,
                    "image_id": image_id,
                    "bbox": bounding_box.tolist(),
                    "category_id": int(category_id),
                    "id": bnd_id,
                    "ignore": 0,
                    "segmentation": segmentation,
                }
                json_dict["annotations"].append(ann)
                bnd_id = bnd_id + 1

        else:
            for obj in get(root, "object"):
                category = get_and_check(obj, "name", 1).text
                if category not in categories:
                    new_id = len(categories)
                    categories[category] = new_id
                category_id = categories[category]
                bndbox = get_and_check(obj, "bndbox", 1)
                xmin = int(get_and_check(bndbox, "xmin", 1).text) - 1
                ymin = int(get_and_check(bndbox, "ymin", 1).text) - 1
                xmax = int(get_and_check(bndbox, "xmax", 1).text)
                ymax = int(get_and_check(bndbox, "ymax", 1).text)
                assert xmax > xmin
                assert ymax > ymin
                o_width = abs(xmax - xmin)
                o_height = abs(ymax - ymin)
                this_mask = mask_cls.copy()
                this_mask = this_mask*0.0
                this_mask[ymin:ymax,xmin:xmax][mask_cls[ymin:ymax,xmin:xmax]==category_id]=1
                this_mask = np.array(this_mask).astype(np.uint8)
                segmentation = binary_mask_to_polygon(this_mask, tolerance=2)
                binary_mask_encoded = mask.encode(np.asfortranarray(this_mask.astype(np.uint8)))
                area = mask.area(binary_mask_encoded)
                if segmentation ==[]:
                    this_mask = mask_cls.copy()
                    this_mask = this_mask*0.0
                    this_mask[ymin:ymax,xmin:xmax]=1
                    this_mask = np.array(this_mask).astype(np.uint8)
                    segmentation = binary_mask_to_polygon(this_mask, tolerance=2)
                if segmentation==[]:
                    continue
                ann = {
                    "area": o_width * o_height,
                    "iscrowd": 0,
                    "image_id": image_id,
                    "bbox": [xmin, ymin, o_width, o_height],
                    "category_id": category_id,
                    "id": bnd_id,
                    "ignore": 0,
                    "segmentation": segmentation,
                }
                json_dict["annotations"].append(ann)
                bnd_id = bnd_id + 1

    for cate, cid in categories.items():
        cat = {"supercategory": "none", "id": cid, "name": cate}
        json_dict["categories"].append(cat)
    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    json_fp = open(json_file, "w")
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()
    print ('-->%d'%now)


if __name__ == "__main__":
    source_dir = './data/VOCdevkit/VOC2012/' #Edit this to your own dataset path.
    xml_dir = source_dir + 'Annotations' #Path of xml data directory.
    train_or_val= 'trainaug' #trainaug|val
    if train_or_val=='val':
        val = True #With GT objects and Masks.
        inst_dir = source_dir +'SegmentationObject'
        mask_dir = source_dir +'SegmentationClass'
    else:
        val = False
        inst_dir = None
        mask_dir = './data/gen_labels/FR_95/mask' #Path of generated pseudo label directory.
    train_list = os.path.join(source_dir+'ImageSets/Segmentation', train_or_val + ".txt") #Path of data list directory.
    train_files = [i.strip() for i in open(train_list) if not i.strip() == ' ']
    json_file = './voc_inst_%s.json'%train_or_val #Save to current dir.
    xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))
    # If you want to do train/test split, you can pass a subset of xml files to convert function.
    print("Number of xml files: {}".format(len(xml_files)))
    convert(xml_files, json_file, mask_dir, train_files, val=val, inst_dir=inst_dir)
    print("Success: {}".format(json_file))
