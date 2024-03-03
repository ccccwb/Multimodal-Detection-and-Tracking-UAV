# voc2coco.py

# pip install lxml

import sys
import os
import json
import cv2
import numpy as np

import xml.etree.ElementTree as ET

from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import Manager

from transforms_coco import (poly2hbb, poly2obb, rectpoly2obb, obb2poly, obb2hbb,
                         hbb2poly, hbb2obb, bbox2type)


START_BOUNDING_BOX_ID = 1
bnd_id = START_BOUNDING_BOX_ID

# PRE_DEFINE_CATEGORIES = {}
# If necessary, pre-define category and its id
PRE_DEFINE_CATEGORIES = {'car':0, 'bus':1, 'truck':2, 'van':3, 'freight_car':4}
                         #  "bottle":5, "bus": 6, "car": 7, "cat": 8, "chair": 9,
                         #  "cow": 10, "diningtable": 11, "dog": 12, "horse": 13,
                         #  "motorbike": 14, "person": 15, "pottedplant": 16,
                         #  "sheep": 17, "sofa": 18, "train": 19, "tvmonitor": 20}
dataset_path = '/media/data3/caiwb/RGBTDetection/data/coco/DV2/val/'
txt_dir = dataset_path + 'vallabel_mod/'
img_dir = dataset_path + 'valimg_mod'
json_file = 'val_mod_illumination.json'

height = 511
width = 639
dark_num = 0
bright_num = 0


def dayOrNight(pic_path):
    global dark_num, bright_num
    gray_img = cv2.imread(pic_path, cv2.IMREAD_GRAYSCALE)
    r,c = gray_img.shape[:2]
    dark_sum=0
    dark_prop=0
    piexs_sum=r*c
    for row in gray_img:
        for colum in row:
            if colum<40:
                dark_sum+=1
    dark_prop=dark_sum/(piexs_sum)
    if dark_prop >=0.1:
        dark_num += 1
        print(pic_path+" is dark!" ) #+ " number is " + str(dark_num)
        return 0
    else:
        bright_num += 1
        print(pic_path+" is bright_num!" ) #+ " number is " + str(bright_num)
        return 1

# def dayOrNight(pic_path):
# 	global dark_num, bright_num
# 	gray_img = cv2.imread(pic_path, cv2.IMREAD_GRAYSCALE)#把图片转换为灰度图
# 	#获取灰度图矩阵的行数和列数
# 	r,c = gray_img.shape[:2]
# 	dark_sum=0	#偏暗的像素 初始化为0个
# 	dark_prop=0	#偏暗像素所占比例初始化为0
# 	piexs_sum=r*c	#整个弧度图的像素个数为r*c
	
# 	#遍历灰度图的所有像素
# 	for row in gray_img:
# 		for colum in row:
# 			if colum<40:	#人为设置的超参数,表示0~39的灰度值为暗
# 				dark_sum+=1
# 	dark_prop=dark_sum/(piexs_sum)	
# 	# print("dark_sum:"+str(dark_sum))
# 	# print("piexs_sum:"+str(piexs_sum))
# 	# print("dark_prop=dark_sum/piexs_sum:"+str(dark_prop))
# 	if dark_prop >=0.1:	#人为设置的超参数:表示若偏暗像素所占比例超过0.78,则这张图被认为整体环境黑暗的图片
# 		dark_num += 1
# 		# print(pic_path+" is dark!" + " number is " + str(dark_num))
# 		return 0
# 		# cv2.imwrite("../DarkPicDir/"+pic,img)#把被认为黑暗的图片保存
# 	else:
# 		bright_num += 1
# 		# print(pic_path+" is bright!" + " number is " + str(bright_num))
# 		return 1

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
def read_file(line, bnd_id):
    line = line.strip()
    # print("Processing %s"%(line))
    txt_f = os.path.join(txt_dir, line)
    image_id = str(get_filename_as_int(line)).zfill(5)
    file_name = str(image_id)+'.jpg'
    img_f = os.path.join(img_dir, file_name)
    illumination =  dayOrNight(img_f)
    image = {'file_name': file_name, 'height': height, 'width': width, 'illumination': illumination,
                'id':image_id}
    file = open(txt_f,'r')  #打开文件
    file_data = file.readlines() #读取所有行
    annotations = []
    for row in file_data:
        poly = []
        tmp_list = row.split(' ') #按‘，’切分每行的数据
        tmp_list[-1] = tmp_list[-1].replace('\n',',') #去掉换行符
        x1, y1, x2, y2, x3, y3, x4, y4, category, _ = tmp_list
        poly.append([x1, y1, x2, y2, x3, y3, x4, y4]) 
        poly = np.array(poly, dtype=np.float32)
        poly = bbox2type(poly, 'obb')
        x, y, w, h, theta = poly[0].tolist()
        if (w <= 0 or h <= 0):
            print(file_name)
            print("w = " + str(w) + ", h = " + str(h))
            continue
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
            print(category)
            new_id = len(categories)
            categories[category] = new_id
        category_id = categories[category]
        ann = {'area': width*height, 'iscrowd': 0, 'image_id':
                image_id, 'bbox':[x, y, w, h, theta],
                'category_id': category_id, 'id': (int)(bnd_id.value), 'ignore': 0,
                'segmentation': []}
        annotations.append(ann)
        bnd_id.value = bnd_id.value + 1
        # print("bnd_id" + str((int)(bnd_id.value)))
    return image, annotations


def get_result(future):
    global json_dict
    image, annotations = future.result()
    json_dict["images"].append(image)
    for ann in annotations:
        json_dict['annotations'].append(ann)


json_dict = {"images":[], "type": "instances", "annotations": [],
                "categories": []}
if __name__ == '__main__':
    list_fp = os.listdir(txt_dir)
    list_fp.sort()
    categories = PRE_DEFINE_CATEGORIES
    # bnd_id = START_BOUNDING_BOX_ID
    executor = ProcessPoolExecutor(15)
    manage = Manager()
    bnd_id = manage.Value(int, START_BOUNDING_BOX_ID)
    future_list = [executor.submit(read_file, line, bnd_id) for line in list_fp] # 提交任务
    for future in as_completed(future_list):
        future.add_done_callback(get_result)

    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()


# if __name__ == '__main__':
#     if len(sys.argv) <= 1:
#         print('2 auguments are need.')
#         print('Usage: %stxt_dir OUTPU_JSON.json'%(sys.argv[0]))
#         exit(1)

    # convert(sys.argv[1], sys.argv[2])
######    python yoyo2coco.py ~/caiwb/RGBTDetection/data/coco/DV2/val/vallabel_mod/ ./test.json