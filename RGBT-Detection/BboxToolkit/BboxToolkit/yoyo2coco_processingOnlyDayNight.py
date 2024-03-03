### 多线程只用来分辨日夜
## 路径设置
dataset_path = '/media/data3/caiwb/RGBTDetection/data/coco/DV2/train/'
txt_dir = dataset_path + 'trainlabelr_mod/'
img_dir = dataset_path + 'trainimg_mod'
json_file = 'trainr_mod_illumination.json'
# pip install lxml

import sys
import os
import json
import cv2
from concurrent.futures import ProcessPoolExecutor,as_completed

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
dark_num = 0
bright_num = 0
def dayOrNight(pic_path):
	global dark_num, bright_num
	gray_img = cv2.imread(pic_path, cv2.IMREAD_GRAYSCALE)#把图片转换为灰度图
	#获取灰度图矩阵的行数和列数
	r,c = gray_img.shape[:2]
	dark_sum=0	#偏暗的像素 初始化为0个
	dark_prop=0	#偏暗像素所占比例初始化为0
	piexs_sum=r*c	#整个弧度图的像素个数为r*c
	
	#遍历灰度图的所有像素
	for row in gray_img:
		for colum in row:
			if colum<40:	#人为设置的超参数,表示0~39的灰度值为暗
				dark_sum+=1
	dark_prop=dark_sum/(piexs_sum)	
	# print("dark_sum:"+str(dark_sum))
	# print("piexs_sum:"+str(piexs_sum))
	# print("dark_prop=dark_sum/piexs_sum:"+str(dark_prop))
	if dark_prop >=0.1:	#人为设置的超参数:表示若偏暗像素所占比例超过0.78,则这张图被认为整体环境黑暗的图片
		dark_num += 1
		print(pic_path+" is dark!" + " number is " + str(dark_num))
		return 0
		# cv2.imwrite("../DarkPicDir/"+pic,img)#把被认为黑暗的图片保存
	else:
		bright_num += 1
		print(pic_path+" is bright!" + " number is " + str(bright_num))
		return 1

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

def thread_day_night(img_f, image_id):
    illumination =  dayOrNight(img_f)
    return image_id, illumination

illumination_dict = {}

def get_result(future):
    global illumination_dict
    image_id, illumination = future.result()
    illumination_dict[image_id] = int(illumination)



def convert(txt_dir, img_dir, json_file):
    global illumination_dict
    list_fp = os.listdir(txt_dir)
    list_fp.sort()
    json_dict = {"images":[], "type": "instances", "annotations": [],
                    "categories": []}    
    categories = PRE_DEFINE_CATEGORIES
    bnd_id = START_BOUNDING_BOX_ID
    height = 511
    width = 639
    executor = ProcessPoolExecutor(40)
    future_list = []
    '''先分辨日夜'''
    for line in list_fp:
        line = line.strip()
        # print("Processing %s"%(line))
        txt_f = os.path.join(txt_dir, line)
        image_id = str(get_filename_as_int(line)).zfill(5)
        file_name = str(image_id)+'.jpg'
        img_f = os.path.join(img_dir, file_name)
        # 多进程分辨日夜
        future_list.append(executor.submit(thread_day_night, img_f, image_id))# 提交任务

    for future in as_completed(future_list):
        future.add_done_callback(get_result) 

    ''''''''''''''''''''
    for line in list_fp:
        line = line.strip()
        # print("Processing %s"%(line))
        txt_f = os.path.join(txt_dir, line)
        image_id = str(get_filename_as_int(line)).zfill(5)
        file_name = str(image_id)+'.jpg'
        img_f = os.path.join(img_dir, file_name)
        image = {'file_name': file_name, 'height': height, 'width': width, 'illumination': illumination_dict[image_id],
            'id':image_id}
        json_dict["images"].append(image)
        file = open(txt_f,'r')  #打开文件
        file_data = file.readlines() #读取所有行
        for row in file_data:
            poly = []
            tmp_list = row.split(' ') #按‘，’切分每行的数据
            # if len(tmp_list) == 10:
            #     continue
            # else:
            #     print(1)
            #     print(tmp_list)
            tmp_list[-1] = tmp_list[-1].replace('\n',',') #去掉换行符
            x1, y1, x2, y2, x3, y3, x4, y4, category, _ = tmp_list
            poly.append([x1, y1, x2, y2, x3, y3, x4, y4]) 
            poly = np.array(poly, dtype=np.float32)
            poly = bbox2type(poly, 'obb')
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
            x, y, w, h, theta = poly[0].tolist()
            if (w <= 0 or h <= 0):
                print(file_name)
                print("w = " + str(w) + ", h = " + str(h))
                continue
            ann = {'area': width*height, 'iscrowd': 0, 'image_id':
                   image_id, 'bbox':[x, y, w, h, theta],
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
    file.close()
if __name__ == '__main__':

    convert(txt_dir, img_dir, json_file)

# if __name__ == '__main__':
#     if len(sys.argv) <= 1:
#         print('2 auguments are need.')
#         print('Usage: %stxt_dir OUTPU_JSON.json'%(sys.argv[0]))
#         exit(1)

    # convert(sys.argv[1], sys.argv[2])
######    python yoyo2coco.py ~/caiwb/RGBTDetection/data/coco/DV2/val/vallabel_mod/ ./test.json