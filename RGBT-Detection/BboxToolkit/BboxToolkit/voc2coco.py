# voc2coco.py

# pip install lxml

import sys
import os
import json
import xml.etree.ElementTree as ET
import numpy as np
from transforms_coco import (poly2hbb, poly2obb, rectpoly2obb, obb2poly, obb2hbb,
                         hbb2poly, hbb2obb, bbox2type)
START_BOUNDING_BOX_ID = 1
PRE_DEFINE_CATEGORIES = {}
# If necessary, pre-define category and its id
PRE_DEFINE_CATEGORIES = {'car':0, 'bus':1, 'truck':2, 'van':3, 'freight_car':4}
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
        raise NotImplementedError('Can not find %s in %s.'%(name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.'%(name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars

def get_bbox(root, name):
    vars = root.findall(name)
    return vars
    
def get_filename_as_int(filename):
    try:
        filename = os.path.splitext(filename)[0]
        return int(filename)
    except:
        raise NotImplementedError('Filename %s is supposed to be an integer.'%(filename))
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def convert(xml_list, xml_dir, json_file):
    list_fp = open(xml_list, 'r')
    json_dict = {"images":[], "type": "instances", "annotations": [],
                 "categories": []}
    categories = PRE_DEFINE_CATEGORIES
    bnd_id = START_BOUNDING_BOX_ID
    for line in list_fp:
        line = line.strip()
        # print("Processing %s"%(line))
        xml_f = os.path.join(xml_dir, line)
        tree = ET.parse(xml_f)
        root = tree.getroot()
        path = get(root, 'path')
        # if len(path) == 1:
        #     filename = os.path.basename(path[0].text)
        #     print("--------------------")
        #     print(filename)
        #     print("--------------------")
        # elif len(path) == 0:
        #     filename = get_and_check(root, 'filename', 1).text
        # else:
        #     raise NotImplementedError('%d paths found in %s'%(len(path), line))
        ## The filename must be a number

        image_id = get_filename_as_int(line)
        size = get_and_check(root, 'size', 1)
        width = int(get_and_check(size, 'width', 1).text)
        height = int(get_and_check(size, 'height', 1).text)
        file_name = str(image_id)+'.jpg'
        image = {'file_name': file_name, 'height': height, 'width': width,
                 'id':image_id}
        json_dict['images'].append(image)
        ## Cruuently we do not support segmentation
        #  segmented = get_and_check(root, 'segmented', 1).text
        #  assert segmented == '0'
        for obj in get(root, 'object'):

            category = get_and_check(obj, 'name', 1).text
            if(category == 'feright_car' or category == 'feright car' or category == 'freight car'):
                print(file_name)
                print(category)
                category = 'freight_car'
            elif(category == 'truvk'):
                print(file_name)
                print(category+"----------------------------------------")
                category = 'truck'
            elif(category == '*'):
                print(file_name)
                print(category+"+++++++++++++++++++")
                continue
            if category not in categories:
                new_id = len(categories)
                categories[category] = new_id
            category_id = categories[category]

            polygon = get_bbox(obj, 'polygon')
            if len(polygon) != 0:      
                polygon = polygon[0]
                x1 = int(get_and_check(polygon, 'x1', 1).text)
                y1 = int(get_and_check(polygon, 'y1', 1).text) 
                x2 = int(get_and_check(polygon, 'x2', 1).text)
                y2 = int(get_and_check(polygon, 'y2', 1).text) 
                x3 = int(get_and_check(polygon, 'x3', 1).text)
                y3 = int(get_and_check(polygon, 'y3', 1).text) 
                x4 = int(get_and_check(polygon, 'x4', 1).text)
                y4 = int(get_and_check(polygon, 'y4', 1).text) 
                poly = []
                poly.append([x1, y1, x2, y2, x3, y3, x4, y4])
                poly = np.array(poly, dtype=np.float32)
                poly = bbox2type(poly, 'obb')
                
            else :
                polygon = get_bbox(obj, 'bndbox')
                if len(polygon) != 0: 
                    polygon = polygon[0]
                    xmin = int(get_and_check(polygon, 'xmin', 1).text)
                    ymin = int(get_and_check(polygon, 'ymin', 1).text) 
                    xmax = int(get_and_check(polygon, 'xmax', 1).text)
                    ymax = int(get_and_check(polygon, 'ymax', 1).text) 
                    # x1 = xmin
                    # y1 = ymin
                    # x2 = xmin
                    # y2 = ymax
                    poly = []
                    poly.append([xmin, ymin, xmax, ymax])
                    poly = np.array(poly, dtype=np.float32)
                    poly = bbox2type(poly, 'obb')
                    
                else :
                    continue
            a,b,c,d,e = poly[0].tolist()
            ann = {'area': width*height, 'iscrowd': 0, 'image_id':
                   image_id, 'bbox':[a,b,c,d,e],
                   'category_id': category_id, 'id': bnd_id, 'ignore': 0,
                   'segmentation': []}
            json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1

    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()
    list_fp.close()


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('3 auguments are need.')
        print('Usage: %s XML_LIST.txt XML_DIR OUTPU_JSON.json'%(sys.argv[0]))
        exit(1)

    convert(sys.argv[1], sys.argv[2], sys.argv[3])

