import json
import shutil
import cv2
import numpy as np
import os
import random
from transforms_coco import (poly2hbb, poly2obb, rectpoly2obb, obb2poly, obb2hbb,
                         hbb2poly, hbb2obb, bbox2type)
# from ..draw import draw_obb
from pycocotools.coco import COCO
id = ['car', 'bus', 'truck', 'van', 'freight_car']
# ---------------------------json画框
def select(json_path, outpath, image_path):
    imgfiles = os.listdir(image_path)   # 读入文件夹
    image_id = len(imgfiles)       # 统计文件夹中的文件个数
    coco = COCO(json_path)
    r = random.randint(0,255)
    g = random.randint(0,255)
    b = random.randint(0,255)

    for i in range(image_id):
        i = i + 1
        # print("{}.jpg".format(str(i).zfill(5)))
        img = cv2.imread(image_path + str(i).zfill(5) + '.jpg')
        annIds = coco.getAnnIds(imgIds=[str(i).zfill(5)], iscrowd=None)
        anns = coco.loadAnns(annIds)
        imgIds = coco.getImgIds(imgIds=[str(i).zfill(5)])
        imgs = coco.loadImgs(imgIds)
        cv2.putText(img, str(imgs[0]['illumination']), (int(imgs[0]['height']/2), int(imgs[0]['height']/2)), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), thickness=2)
        for n in range(len(anns)):
            x, y, w, h, theta = anns[n]['bbox']
            rectangle = [x, y, w, h, theta]
            rectangle = np.array(rectangle, dtype=np.float32)
            poly = bbox2type(rectangle, 'poly')
            # x1 = int(np.cos(rectangle[4])*(-rectangle[2]/2) - np.sin(rectangle[4])*(-rectangle[3]/2) + rectangle[0])
            # x2 = int(np.cos(rectangle[4])*(rectangle[2]/2) - np.sin(rectangle[4])*(-rectangle[3]/2) + rectangle[0])
            # x3 = int(np.cos(rectangle[4])*(-rectangle[2]/2) - np.sin(rectangle[4])*(rectangle[3]/2) + rectangle[0])
            # x4 = int(np.cos(rectangle[4])*(rectangle[2]/2) - np.sin(rectangle[4])*(rectangle[3]/2) + rectangle[0])
            # y1 = int(np.sin(rectangle[4])*(-rectangle[2]/2) + np.cos(rectangle[4])*(-rectangle[3]/2) + rectangle[1])
            # y2 = int(np.sin(rectangle[4])*(rectangle[2]/2) + np.cos(rectangle[4])*(-rectangle[3]/2) + rectangle[1])
            # y3 = int(np.sin(rectangle[4])*(-rectangle[2]/2) + np.cos(rectangle[4])*(rectangle[3]/2) + rectangle[1])
            # y4 = int(np.sin(rectangle[4])*(rectangle[2]/2) + np.cos(rectangle[4])*(rectangle[3]/2) + rectangle[1])
            x1 = poly[0]
            y1 = poly[1]
            x2 = poly[2]
            y2 = poly[3]
            x3 = poly[4]
            y3 = poly[5]
            x4 = poly[6]
            y4 = poly[7]
            color = (0, 0, 255)
            line_thickness = 1
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, line_thickness, 4)
            cv2.line(img, (int(x2), int(y2)), (int(x3), int(y3)), color, line_thickness, 4)
            cv2.line(img, (int(x3), int(y3)), (int(x4), int(y4)), color, line_thickness, 4)
            cv2.line(img, (int(x4), int(y4)), (int(x1), int(y1)), color, line_thickness, 4)
            cv2.putText(img, id[int(anns[n]['category_id'])], (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, thickness=line_thickness)
        print(outpath + str(i).zfill(5) + '.jpg')
        cv2.imwrite(outpath + str(i).zfill(5) + '.jpg', img)
    # coco = COCO(json_path)
    # json_file = open(json_path, 'r')
    # infos = json.load(json_file)
    # ann_ids = coco.get_ann_ids(img_ids=[img_id])
    # ann_info = coco.load_anns(ann_ids) 
    # for i in infos:
    #     images = i["image_id"]
        # if i['score'] >= 0.5:
        #     if i['image_id']!= images:
        #         img = cv2.imread(image_path + images + '.jpg')
        #     else:
        #         # 换成你自己的类别
        #         img = cv2.imread(outpath + images + '.jpg')
        #     xc, yc = int(i['bbox'][0]), int(i['bbox'][1])
        #     w, h = int(i['bbox'][2]), int(i['bbox'][3])
        #     theta = i['bbox'][4]
        #     rect = cv2.boxPoints((xc,yc),(w,h),theta)
        #     # x2, y2 = x1 + w, y1 + h
        #     # img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), thickness=1)
        #     box = np.int0(rect)
        #     # 画出来
        #     cv2.drawContours(img, [box], 0, (0, 0, 255), 5)
        #     img_name = outpath + images + '.jpg'
        #     # import pdb
        #     # pdb.set_trace()
        #     cv2.imwrite(img_name, img)

# def showNimages(imageidFile, annFile, imageFile, resultFile):
#     """
#     :param imageidFile: 要查看的图片imageid，存储一列在csv文件里 （目前设计的imageid需要为6位数，如果少于6位数，可以在前面加多个0）
#     :param annFile:使用的标注文件
#     :param imageFile:要读取的image所在文件夹
#     :param resultFile:画了标注之后的image存储文件夹
#     :return:
#     """
    # data = pd.read_csv(imageidFile)
    # list = data.values.tolist()
    # image_id = []  # 存储的是要提取图片id
    # for i in range(len(list)):
    #     image_id.append(list[i][0])
    # print(image_id)
    # print(len(image_id))
    # coco = COCO(annFile)
 
    # for i in range(len(image_id)):
    #     image = cv2.imread(imageFile + '000000' + str(image_id[i]) + '.jpg')
    #     annIds = coco.getAnnIds(imgIds=image_id[i], iscrowd=None)
    #     anns = coco.loadAnns(annIds)
    #     for n in range(len(anns)):
    #         x, y, w, h = anns[n]['bbox']
    #         x, y, w, h = int(x), int(y), int(w), int(h)
    #         # print(x, y, w, h)
    #         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255))
    #     cv2.imwrite(resultFile + '000000' + str(image_id[i]) + 'result.png', image)

if __name__ == "__main__":
    json_path = "/media/data3/caiwb/RGBTDetection/data/coco/annotations/val_mod_illumination.json"
    # json_path_r = "/media/data3/caiwb/RGBTDetection/data/coco/annotations/valr_mod.json"

    out_path = "/media/data3/caiwb/RGBTDetection/data/coco/gt_valmod/"
    image_path = "/media/data3/caiwb/RGBTDetection/data/coco/DV2/val/valimg_mod/"


    # json_path = "/media/data3/caiwb/RGBTDetection/data/coco/annotations/val_mod_illumination.json"
    # # json_path_r = "/media/data3/caiwb/RGBTDetection/data/coco/annotations/valr_mod.json"

    # out_path = "/media/data3/caiwb/RGBTDetection/data/coco/gt_valmod/"
    # image_path = "/media/data3/caiwb/RGBTDetection/data/coco/gt_valmod/"  
    select(json_path, out_path, image_path)
    # select(json_path_r, out_path, out_path)
