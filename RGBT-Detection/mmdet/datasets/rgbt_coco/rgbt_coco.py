from ..builder import DATASETS
from ..pipelines import Compose
from .rgbt_custom import RgbtCustomDataset

import os.path as osp
import numpy as np

from pycocotools.coco import COCO

@DATASETS.register_module()
class RgbtCocoDataset(RgbtCustomDataset):
    CLASSES = ('car', 'bus', 'truck', 'van', 'freight_car')   
    def load_annotations_r(self, ann_file):
        self.coco_r = COCO(ann_file)
        self.cat_ids = self.coco_r.getCatIds(catNms=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids_r = self.coco_r.getImgIds()

        img_infos_r = []
        for i in self.img_ids_r:
            info_r = self.coco_r.loadImgs([i])[0]
            info_r['filename'] = info_r['file_name']
            img_infos_r.append(info_r)
        #print(ann_file)
        return img_infos_r

    def load_annotations_i(self, ann_file):
        self.coco_i = COCO(ann_file)
        self.cat_ids = self.coco_i.getCatIds(catNms=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids_i = self.coco_i.getImgIds()

        img_infos_i = []
        for i in self.img_ids_i:
            info_i = self.coco_i.loadImgs([i])[0]
            info_i['filename'] = info_i['file_name']
            img_infos_i.append(info_i)
        return img_infos_i

    def get_ann_info_r(self, idx):
        img_id_r = self.img_infos_r[idx]['id']           # rgb
        ann_ids_r = self.coco_r.getAnnIds(imgIds=[img_id_r])
        ann_info_r = self.coco_r.loadAnns(ann_ids_r)
        # print(ann_info_r)
        return self._parse_ann_info_r(ann_info_r)

    def get_ann_info_i(self, idx):
        img_id_i = self.img_infos_i[idx]['id']           # infrared
        ann_ids_i = self.coco_i.getAnnIds(imgIds=[img_id_i])
        ann_info_i = self.coco_i.loadAnns(ann_ids_i)
        # print(ann_info_i)
        return self._parse_ann_info_i(ann_info_i)

    def _parse_ann_info_r(self, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            # if ann.get('ignore', False):
            #     continue

            x, y, w, h, a = ann['bbox']
            # inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            # inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            # if inter_w * inter_h == 0:
            #     continue
            if ann['area'] <= 0:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x, y, w, h, a]
            if w <= 0 or h <= 0:
                continue
            # print(bbox)
            # bbox = [x1, y1, x1 + w, y1 + h]
            # if ann.get('iscrowd', False):
            #     gt_bboxes_ignore.append(bbox)
            # else:
            gt_bboxes.append(bbox)
            gt_labels.append(self.cat2label[ann['category_id']])
            gt_masks_ann.append(ann['segmentation'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 5), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 5), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann)

        return ann
    def _parse_ann_info_i(self, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            # if ann.get('ignore', False):
            #     continue

            x, y, w, h, a = ann['bbox']
            
            # inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            # inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            # if inter_w * inter_h == 0:
            #     continue
            if ann['area'] <= 0:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x, y, w, h, a]
            # print(bbox)
            # bbox = [x1, y1, x1 + w, y1 + h]
            # if ann.get('iscrowd', False):
            #     gt_bboxes_ignore.append(bbox)
            # else:
            gt_bboxes.append(bbox)
            gt_labels.append(self.cat2label[ann['category_id']])
            gt_masks_ann.append(ann['segmentation'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 5), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 5), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann)

        return ann