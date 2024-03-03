import os.path as osp
import os

import mmcv
import numpy as np
from torch.utils.data import Dataset

from mmdet.core import eval_map, eval_recalls
from ..builder import DATASETS
from ..pipelines import Compose
from mmdet.core import eval_arb_map, eval_arb_recalls
from mmdet.core import eval_recalls
import BboxToolkit as bt

@DATASETS.register_module()
class RgbtCustomDataset(Dataset):
    """Custom dataset for detection.

    The annotation format is shown as follows. The `ann` field is optional for
    testing.

    .. code-block:: none

        [
            {
                'filename': 'a.jpg',
                'width': 1280,
                'height': 720,
                'ann': {
                    'bboxes': <np.ndarray> (n, 4),
                    'labels': <np.ndarray> (n, ),
                    'bboxes_ignore': <np.ndarray> (k, 4), (optional field)
                    'labels_ignore': <np.ndarray> (k, 4) (optional field)
                }
            },
            ...
        ]

    Args:
        ann_file (str): Annotation file path.
        pipeline (list[dict]): Processing pipeline.
        classes (str | Sequence[str], optional): Specify classes to load.
            If is None, ``cls.CLASSES`` will be used. Default: None.
        data_root (str, optional): Data root for ``ann_file``,
            ``img_prefix``, ``seg_prefix``, ``proposal_file`` if specified.
        test_mode (bool, optional): If set True, annotation will not be loaded.
        filter_empty_gt (bool, optional): If set true, images without bounding
            boxes will be filtered out.
    """

    CLASSES = None

    def __init__(self,
                 ann_file_r=None,
                 ann_file_i=None,
                 pipeline=None,
                 classes=None,
                 data_root=None,
                 img_prefix_r='',
                 img_prefix_i='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True):
        # prefix of ann path
        self.ann_file_r = ann_file_r # 可见光标注
        self.ann_file_i = ann_file_i # 红外标注

        # prefix of images path
        self.img_prefix_r = img_prefix_r
        self.img_prefix_i = img_prefix_i

        self.data_root = data_root
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.CLASSES = self.get_classes(classes)
        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file_r):
                self.ann_file_r = osp.join(self.data_root, self.ann_file_r)
            if not osp.isabs(self.ann_file_i):
                self.ann_file_i = osp.join(self.data_root, self.ann_file_i)
            if not (self.img_prefix_r is None or osp.isabs(self.img_prefix_r)):
                self.img_prefix_r = osp.join(self.data_root, self.img_prefix_r)
            if not (self.img_prefix_i is None or osp.isabs(self.img_prefix_i)):
                self.img_prefix_i = osp.join(self.data_root, self.img_prefix_i)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.proposal_file is None
                    or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root,
                                              self.proposal_file)

        # load annotations (and proposals)
        self.img_infos_r = self.load_annotations_r(self.ann_file_r)
        self.img_infos_i = self.load_annotations_i(self.ann_file_i)


        # filter data infos if classes are customized
        if self.custom_classes:
            self.img_infos_r = self.get_subset_by_classes()
            self.img_infos_i = self.get_subset_by_classes()


        if self.proposal_file is not None:
            self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None
        # filter images too small
        if not test_mode:
            valid_inds_rgbt = self._filter_imgs_rgbt()
            self.img_infos_r = [self.img_infos_r[i] for i in valid_inds_rgbt]

            self.img_infos_i = [self.img_infos_i[i] for i in valid_inds_rgbt]

            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds_rgbt]
        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        # processing pipeline
        self.pipeline = Compose(pipeline)

    def __len__(self):
        return len(self.img_infos_i)             #infrared

    def load_annotations_r(self, ann_file):         # rgb
        return mmcv.load(ann_file)

    def load_annotations_i(self, ann_file):       # infrared
        return mmcv.load(ann_file)

    def load_proposals(self, proposal_file):
        return mmcv.load(proposal_file)

    def get_ann_info_r(self, idx):                    # rgb
        return self.img_infos_r[idx]['ann']

    def get_ann_info_i(self, idx):                    # infrared
        return self.img_infos_i[idx]['ann']

    def _filter_imgs_r(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds_r = []
        ids_with_ann_r = set(_['image_id'] for _ in self.coco_r.anns.values())
        for i, img_info in enumerate(self.img_infos_r):            # rgb
            if self.img_ids_r[i] not in ids_with_ann_r:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds_r.append(i)
        print('RGB:',len(valid_inds_r))
        return valid_inds_r

    def _filter_imgs_i(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds_i = []
        ids_with_ann_i = set(_['image_id'] for _ in self.coco_i.anns.values())
        for i, img_info in enumerate(self.img_infos_i):            # infrared
            if self.img_ids_i[i] not in ids_with_ann_i:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds_i.append(i)
        print('tir:',len(valid_inds_i))
        return valid_inds_i

    def _filter_imgs_rgbt(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds_rgbt = []
        ids_with_ann_r = set(_['image_id'] for _ in self.coco_r.anns.values()) #获得ann标注文件里的image_id的值
        ids_with_ann_i = set(_['image_id'] for _ in self.coco_i.anns.values()) #获得ann标注文件里的image_id的值
        for i, img_info in enumerate(self.img_infos_r):     #去除无标注文件的图片
            if (self.img_ids_r[i] not in ids_with_ann_r):
                continue
            if (self.img_ids_i[i] not in ids_with_ann_i):
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds_rgbt.append(i)
        print('RGBT:',len(valid_inds_rgbt))
        return valid_inds_rgbt

    def pre_pipeline(self, results):
        #"""Prepare results dict for pipeline"""
        results['img_prefix_r'] = self.img_prefix_r
        results['img_prefix_i'] = self.img_prefix_i


        results['proposal_file'] = self.proposal_file

        results['bbox_fields_r'] = []
        results['bbox_fields_i'] = []

        results['mask_fields_r'] = []
        results['mask_fields_i'] = []
        
        results['seg_prefix'] = self.seg_prefix
        results['seg_fields'] = []
        
        results['cls'] = self.CLASSES

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.img_infos_i[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1
    def _rand_another(self, idx):
        """Get another random index from the same group as the given index"""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        """Get training/test data after pipeline

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                True).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """
        img_info_r = self.img_infos_r[idx]
        img_info_i = self.img_infos_i[idx]

        ann_info_r = self.get_ann_info_r(idx)
        ann_info_i = self.get_ann_info_i(idx)
        results = dict(img_info_r=img_info_r, img_info_i=img_info_i, ann_info_r=ann_info_r, ann_info_i=ann_info_i)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data  after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys intorduced by
                piepline.
        """

        img_info_r = self.img_infos_r[idx]
        img_info_i = self.img_infos_i[idx]
        results = dict(img_info_r=img_info_r, img_info_i = img_info_i)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    @classmethod
    def get_classes(cls, classes=None):
        """Get class names of current dataset

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        """
        if classes is None:
            cls.custom_classes = False
            return cls.CLASSES

        cls.custom_classes = True
        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        return class_names

    def get_subset_by_classes(self):
        return self.img_infos_i ########

    def format_results(self,
                       results,
                       with_merge=True,
                       ign_scale_ranges=None,
                       iou_thr=0.5,
                       nproc=4,
                       save_dir=None,
                       **kwargs):
        nproc = min(nproc, os.cpu_count())
        if mmcv.is_list_of(results, tuple):
            dets, segments = results
        else:
            dets = results

        if not with_merge:
            results = [(img_info_i['id'], result)
                       for img_info_i, result in zip(self.img_infos_i, results)]
            if save_dir is not None:
                id_list, dets_list = zip(*results)
                bt.save_dota_submission(save_dir, id_list, dets_list, 'Task1', self.CLASSES)
            return results

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 with_merge=False,
                 ign_diff=True,
                 ign_scale_ranges=None,
                 save_dir=None,
                 merge_iou_thr=0.1,
                 use_07_metric=True,
                 scale_ranges=None,
                 eval_iou_thr=[0.5],
                 proposal_nums=(2000, ),
                 nproc=10):
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        

        eval_results = {}
        if metric == 'mAP':
            # print("results")
            # print(results)
            merged_results = self.format_results(
                results,
                nproc=nproc,
                with_merge=with_merge,
                ign_scale_ranges=ign_scale_ranges,
                iou_thr=merge_iou_thr,
                save_dir=save_dir)
            # print("merged_results")
            # print(merged_results)
            infos = self.ori_infos if with_merge else self.img_infos_i ########
            id_mapper = {ann['id']: i for i, ann in enumerate(infos)}
            det_results, annotations = [], []
            for k, v in merged_results:
                det_results.append(v)
                ann = self.get_ann_info_i(id_mapper[k]) #infos[id_mapper[k]]['ann']
                # print(ann)
                gt_bboxes = ann['bboxes']
                gt_labels = ann['labels']
                diffs = ann.get(
                    'diffs', np.zeros((gt_bboxes.shape[0], ), dtype=np.int))

                gt_ann = {}
                if ign_diff:
                    gt_ann['bboxes_ignore'] = gt_bboxes[diffs == 1]
                    gt_ann['labels_ignore'] = gt_labels[diffs == 1]
                    gt_bboxes = gt_bboxes[diffs == 0]
                    gt_labels = gt_labels[diffs == 0]
                gt_ann['bboxes'] = gt_bboxes
                gt_ann['labels'] = gt_labels
                annotations.append(gt_ann)

            print('\nStart calculate mAP!!!')
            print('Result is Only for reference,',
                  'final result is subject to DOTA_devkit')
            mean_ap, _ = eval_arb_map(
                det_results,
                annotations,
                scale_ranges=scale_ranges,
                iou_thr=eval_iou_thr,
                use_07_metric=use_07_metric,
                dataset=self.CLASSES,
                logger=logger,
                nproc=nproc)
            eval_results['mAP'] = mean_ap
        elif metric == 'recall':
            assert mmcv.is_list_of(results, np.ndarray)
            gt_bboxes = []
            for info in self.img_infos_i: ########
                bboxes = info['ann']['bboxes']
                if ign_diff:
                    diffs = info['ann'].get(
                        'diffs', np.zeros((bboxes.shape[0], ), dtype=np.int))
                    bboxes = bboxes[diffs == 0]
                gt_bboxes.append(bboxes)
            if isinstance(eval_iou_thr, float):
                eval_iou_thr = [eval_iou_thr]
            recalls = eval_arb_recalls(
                gt_bboxes, results, True, proposal_nums, eval_iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(eval_iou_thr):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results