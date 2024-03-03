from pickle import NONE
from numpy.lib.function_base import average
import BboxToolkit as bt

import cv2
import mmcv
import warnings
import itertools
import numpy as np
import pycocotools.mask as maskUtils
import os.path as osp
import torch
from mmdet.core import bbox2type, get_bbox_areas

from mmcv.parallel import DataContainer as DC
from mmdet.core import PolygonMasks, BitmapMasks
from mmdet.datasets.builder import PIPELINES
from ..obb.misc import bbox2mask, switch_mask_type, mask2bbox, rotate_polygonmask

from ..loading import LoadAnnotations
from ..formating import DefaultFormatBundle, Collect, to_tensor
from ..transforms import RandomFlip
from ..compose import Compose
from ..transforms import Resize, Normalize, Pad
from ..obb.base import RandomOBBRotate
@PIPELINES.register_module()
class LoadRGBTImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix_r" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img_r", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes()`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)


        #RGB图像
        if results['img_prefix_r'] is not None:
            filename_r = osp.join(results['img_prefix_r'],
                                results['img_info_r']['filename'])
        else:
            filename_r = results['img_info_r']['filename']
        img_bytes_r = self.file_client.get(filename_r)
        img_r = mmcv.imfrombytes(img_bytes_r, flag=self.color_type)
        if self.to_float32:
            img_r = img_r.astype(np.float32)

        #红外图像
        if results['img_prefix_i'] is not None:
            filename_i = osp.join(results['img_prefix_i'],
                                results['img_info_i']['filename'])
        else:
            filename_i = results['img_info_i']['filename']
        img_bytes_i = self.file_client.get(filename_i)
        img_i = mmcv.imfrombytes(img_bytes_i, flag=self.color_type)   #红外图像为灰度
        # if self.to_float32:
        #     img_i = img_i.astype(np.float32)

        results['filename_r'] = filename_r
        results['filename_i'] = filename_i
        results['img_r'] = img_r
        results['img_i'] = img_i

        results['img_fields'] = ['img_r', 'img_i']

        results['ori_filename'] = results['img_info_r']['filename']

        results['img_shape'] = img_r.shape
        results['ori_shape'] = img_r.shape

        results['illumination'] = results['img_info_r']['illumination']

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str
########################################计算旋转框交并比########################################
import math
from shapely.geometry import Polygon

def myobb2poly(obbox):
    x, y, w, h, theta = obbox
    center = x, y
    Cos, Sin = math.cos(theta), math.sin(theta)

    vector1 = w/2 * Cos, -w/2 * Sin
    vector2 = -h/2 * Sin, -h/2 * Cos

    point1 = center[0]+vector1[0]+vector2[0], center[1]+vector1[1]+vector2[1]
    point2 = center[0]+vector1[0]-vector2[0], center[1]+vector1[1]-vector2[1]
    point3 = center[0]-vector1[0]-vector2[0], center[1]-vector1[1]-vector2[1]
    point4 = center[0]-vector1[0]+vector2[0], center[1]-vector1[1]+vector2[1]
    return [point1, point2, point3, point4]

def intersection(g, p):
    g, p = myobb2poly(g), myobb2poly(p)
    g=np.asarray(g)
    p=np.asarray(p)
    g = Polygon(g[:8].reshape((4, 2)))
    p = Polygon(p[:8].reshape((4, 2)))
    if not g.is_valid or not p.is_valid:
        return 0
    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if union == 0:
        return 0
    else:
        return inter/union       
########################################################################################


@PIPELINES.register_module()
class LoadRGBTAnnotations(LoadAnnotations):

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_seg=False,
                 with_poly_as_mask=True,
                 poly2mask=False,
                 file_client_args=dict(backend='disk')):
        self.with_bbox = with_bbox
        self.with_poly_as_mask = with_poly_as_mask
        self.with_label = with_label
        self.with_mask = False
        self.with_seg = with_seg
        self.poly2mask = poly2mask
        self.file_client_args = file_client_args.copy()
        self.file_client = None
    def __call__(self, results):
        """Call function to load multiple types annotations

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """
        results['bbox_fields'] = []
        results['mask_fields'] = []
        if self.with_bbox:
            results = self._load_bboxes_r(results)
            results = self._load_bboxes_i(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels_r(results)
            results = self._load_labels_i(results)
        if self.with_mask:
            results = self._load_masks_r(results)
            results = self._load_masks_i(results)
        # if self.with_seg:
        #     results = self._load_semantic_seg(results)
        return results

    def _load_bboxes_r(self, results):
        ann_info = results['ann_info_r']
        gt_bboxes = ann_info['bboxes'].copy()
        if(gt_bboxes.shape[-1] == 0):
            results['gt_bboxes_r'] = np.array(None, dtype = float)
        else:
            results['gt_bboxes_r'] = bt.bbox2type(gt_bboxes, 'hbb')
        results['bbox_fields'].append('gt_bboxes_r')
        if self.with_poly_as_mask:
            if(gt_bboxes.shape[-1] == 0):
                results['gt_masks_r'] = np.array(None, dtype = float)
            else:
                h, w = results['img_info_r']['height'], results['img_info_r']['width']
                polys = bt.bbox2type(gt_bboxes.copy(), 'poly')
                mask_type = 'bitmap' if self.poly2mask else 'polygon'
                gt_masks = poly2mask(polys, w, h, mask_type)
                results['gt_masks_r'] = gt_masks
            results['mask_fields'].append('gt_masks_r')
        return results
           
    def _load_bboxes_i(self, results):
        ann_info = results['ann_info_i']
        gt_bboxes = ann_info['bboxes'].copy()
        if(gt_bboxes.shape[-1] == 0):
            results['gt_bboxes_i'] = np.array(None, dtype = float)
        else:
            results['gt_bboxes_i'] = bt.bbox2type(gt_bboxes, 'hbb')
        results['bbox_fields'].append('gt_bboxes_i')
        if self.with_poly_as_mask:
            if(gt_bboxes.shape[-1] == 0):
                results['gt_masks_i'] = np.array(None, dtype = float)
            else:
                h, w = results['img_info_i']['height'], results['img_info_i']['width']
                polys = bt.bbox2type(gt_bboxes.copy(), 'poly')
                mask_type = 'bitmap' if self.poly2mask else 'polygon'
                gt_masks = poly2mask(polys, w, h, mask_type)
                results['gt_masks_i'] = gt_masks
            results['mask_fields'].append('gt_masks_i')
        return results

    def _load_labels_r(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """
        results['gt_labels_r'] = results['ann_info_r']['labels'].copy()
        return results
    def _load_labels_i(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """
        results['gt_labels_i'] = results['ann_info_i']['labels'].copy()
        return results

    def _load_masks_r(self, results):
        """Private function to load mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded mask annotations.
                If ``self.poly2mask`` is set ``True``, `gt_mask` will contain
                :obj:`PolygonMasks`. Otherwise, :obj:`BitmapMasks` is used.
        """

        h, w = results['img_info_r']['height'], results['img_info_r']['width']
        gt_masks = results['ann_info_r']['masks']
        if self.poly2mask:
            gt_masks = BitmapMasks(
                [self._poly2mask(mask, h, w) for mask in gt_masks], h, w)
        else:
            gt_masks = PolygonMasks(
                [self.process_polygons(polygons) for polygons in gt_masks], h,
                w)
        results['gt_masks_r'] = gt_masks
        results['mask_fields'].append('gt_masks_r')
        return results

    def _load_masks_i(self, results):
        """Private function to load mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded mask annotations.
                If ``self.poly2mask`` is set ``True``, `gt_mask` will contain
                :obj:`PolygonMasks`. Otherwise, :obj:`BitmapMasks` is used.
        """

        h, w = results['img_info_i']['height'], results['img_info_i']['width']
        gt_masks = results['ann_info_i']['masks']
        if self.poly2mask:
            gt_masks = BitmapMasks(
                [self._poly2mask(mask, h, w) for mask in gt_masks], h, w)
        else:
            gt_masks = PolygonMasks(
                [self.process_polygons(polygons) for polygons in gt_masks], h,
                w)
        results['gt_masks_i'] = gt_masks
        results['mask_fields'].append('gt_masks_i')
        return results        

def poly2mask(polys, w, h, mask_type='polygon'):
    assert mask_type in ['polygon', 'bitmap']
    if mask_type == 'bitmap':
        masks = []
        for poly in polys:
            rles = maskUtils.frPyObjects([poly.tolist()], h, w)
            masks.append(maskUtils.decode(rles[0]))
        gt_masks = BitmapMasks(masks, h, w)

    else:
        gt_masks = PolygonMasks([[poly] for poly in polys], h, w)
    return gt_masks

def mask2obb(gt_masks):
    obboxes = []
    if isinstance(gt_masks, PolygonMasks):
        for mask in gt_masks.masks:
            all_mask_points = np.concatenate(mask, axis=0)[None, ...]
            obboxes.append(bt.bbox2type(all_mask_points, 'obb'))
    elif isinstance(gt_masks, BitmapMasks):
        for mask in gt_masks.masks:
            try:
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            except ValueError:
                _, contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            max_contour = max(contours, key=len).reshape(1, -1)
            obboxes.append(bt.bbox2type(max_contour, 'obb'))
    else:
        raise NotImplementedError

    if not obboxes:
        return np.zeros((0, 5), dtype=np.float32)
    else:
        obboxes = np.concatenate(obboxes, axis=0)
        return obboxes
def mask2poly(gt_masks):
    polys = []
    if isinstance(gt_masks, PolygonMasks):
        for mask in gt_masks.masks:
            if len(mask) == 1 and mask[0].size == 8:
                polys.append(mask)
            else:
                all_mask_points = np.concatenate(mask, axis=0)[None, ...]
                obbox = bt.bbox2type(all_mask_points, 'obb')
                polys.append(bt.bbox2type(obbox, 'poly'))
    elif isinstance(gt_masks, BitmapMasks):
        for mask in gt_masks.masks:
            try:
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            except ValueError:
                _, contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            max_contour = max(contours, key=len).reshape(1, -1)
            obbox = bt.bbox2type(max_contour, 'obb')
            polys.append(bt.bbox2type(obbox, 'poly'))
    else:
        raise NotImplementedError

    if not polys:
        return np.zeros((0, 8), dtype=np.float32)
    else:
        polys = np.concatenate(polys, axis=0)
        return polys
@PIPELINES.register_module()
class RGBTFliterEmpty:

    def __call__(self, results):
        num_objs = 0
        for k in ['gt_bboxes_r', 'gt_masks_r', 'gt_labels_r', 'gt_bboxes_i', 'gt_masks_i', 'gt_labels_i']:
            if k in results:
                num_objs += len(results[k])
        if num_objs == 0:
            return None

        return results

@PIPELINES.register_module()
class RGBTMask2OBB(object):

    def __init__(self,
                 mask_keys=['gt_masks_r', 'gt_masks_ignore', 'gt_masks_i'],
                 obb_keys=['gt_obboxes_r', 'gt_obboxes_ignore', 'gt_obboxes_i'],
                 obb_type='obb'):
        assert len(mask_keys) == len(obb_keys)
        assert obb_type in ['obb', 'poly']
        self.mask_keys = mask_keys
        self.obb_keys = obb_keys
        self.obb_type = obb_type

    def __call__(self, results):
        trans_func = mask2obb if self.obb_type == 'obb' else mask2poly
        for mask_k, obb_k in zip(self.mask_keys, self.obb_keys):
            if mask_k in results:
                mask = results[mask_k]
                obb = trans_func(mask)
                results[obb_k] = obb
        return results

@PIPELINES.register_module()
class RGBTDefaultFormatBundle(DefaultFormatBundle):

    def __call__(self, results):
        if 'img_r' in results:
            img_r = results['img_r']
            img_i = results['img_i']
            # add default meta keys
            results = self._add_default_meta_keys(results)
            if len(img_r.shape) < 3:
                img_r = np.expand_dims(img_r, -1)
            if len(img_i.shape) < 3:
                img_i = np.expand_dims(img_i, -1)
            img_r = np.ascontiguousarray(img_r.transpose(2, 0, 1))
            img_i = np.ascontiguousarray(img_i.transpose(2, 0, 1))
            results['img_r'] = DC(to_tensor(img_r), stack=True)
            results['img_i'] = DC(to_tensor(img_i), stack=True)
        for key in ['proposals', 'gt_bboxes_r', 'gt_bboxes_ignore', 'gt_bboxes_i', 
                    'gt_obboxes_r', 'gt_obboxes_ignore', 'gt_obboxes_i', 'gt_labels_r', 'gt_labels_i']:
            if key not in results:
                continue
            if key in ['gt_obboxes_r', 'gt_obboxes_i'] + results.get('bbox_', []):
                results[key] = results[key].astype(np.float32)
            results[key] = DC(to_tensor(results[key]))
        if 'gt_masks_r' in results:
            results['gt_masks_r'] = DC(results['gt_masks_r'], cpu_only=True)
        if 'gt_masks_i' in results:
            results['gt_masks_i'] = DC(results['gt_masks_i'], cpu_only=True)
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None, ...]), stack=True)

        return results
    def _add_default_meta_keys(self, results):
        """Add default meta keys.

        We set default meta keys including `pad_shape`, `scale_factor` and
        `img_norm_cfg` to avoid the case where no `Resize`, `Normalize` and
        `Pad` are implemented during the whole pipeline.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            results (dict): Updated result dict contains the data to convert.
        """
        img_r = results['img_r']
        img_i = results['img_i']
        results.setdefault('pad_shape', img_r.shape)
        results.setdefault('scale_factor', 1.0)
        num_channels = 1 if len(img_r.shape) < 3 else img_r.shape[2]
        results.setdefault(
            'img_norm_cfg',
            dict(
                mean=np.zeros(num_channels, dtype=np.float32),
                std=np.ones(num_channels, dtype=np.float32),
                to_rgb=False))
        return results
@PIPELINES.register_module()
class RGBTResize(Resize):
    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        for key in results.get('img_fields', ['img_r']):
            if self.keep_ratio:
                img_i, scale_factor = mmcv.imrescale(
                    results['img_i'], results['scale'], return_scale=True)
                img_r, scale_factor = mmcv.imrescale(
                    results[key], results['scale'], return_scale=True)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img_r.shape[:2]
                h, w = results[key].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img_i, w_scale, h_scale = mmcv.imresize(
                    results['img_i'], results['scale'], return_scale=True)
                img_r, w_scale, h_scale = mmcv.imresize(
                    results[key], results['scale'], return_scale=True)

            results[key] = img_r
            results['img_i'] = img_i
            # cv2.imwrite('1.jpg',img_r)
            # cv2.imwrite('2.jpg',img_i)
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
            results['img_shape'] = img_r.shape
            # in case that there is no padding
            results['pad_shape'] = img_r.shape
            results['scale_factor'] = scale_factor
            results['keep_ratio'] = self.keep_ratio

    def _resize_bboxes(self, results):
        """Resize bounding boxes with ``results['scale_factor']``."""
        img_shape = results['img_shape']
        for key in results.get('bbox_fields', []):
            # print("results[key]" + str(results[key]))
            # print("results['scale_factor']" + str(results['scale_factor']))
            bboxes = results[key] * results['scale_factor']
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            results[key] = bboxes

@PIPELINES.register_module()
class RGBTNormalize(Normalize):
    def __call__(self, results):
        # """Call function to normalize images.

        # Args:
        #     results (dict): Result dict from loading pipeline.

        # Returns:
        #     dict: Normalized results, 'img_norm_cfg' key is added into
        #         result dict.

        # """
        for key in results.get('img_fields', ['img_r', 'img_i']):
            results[key] = mmcv.imnormalize(results[key], self.mean, self.std,
                                            self.to_rgb)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)

        return results
@PIPELINES.register_module()
class RGBTPad(Pad):
    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        for key in results.get('img_fields', ['img_r','img_i']):
            if self.size is not None:
                padded_img = mmcv.impad(results[key], self.size, self.pad_val)
            elif self.size_divisor is not None:
                padded_img = mmcv.impad_to_multiple(
                    results[key], self.size_divisor, pad_val=self.pad_val)
            results[key] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

@PIPELINES.register_module()
class RandomRGBTRotate(RandomOBBRotate):
    def __call__(self, results):
        results['rotate_after_flip'] = self.rotate_after_flip
        if 'angle' not in results:
            results['angle'] = self.get_random_angle(results)
        if results['angle'] == 0:
            results['matrix'] = np.eye(3)
            return results
        matrix, w, h = self.get_matrix_and_size(results)
        results['matrix'] = matrix
        img_bound = np.array([[0, 0, w, 0, w, h, 0, h]])
        self.base_rotate(results, matrix, w, h, img_bound)

        for k in results.get('img_fields', []):
            if k != 'img_r' or k != 'img_i':
                results[k] = cv2.warpAffine(results[k], matrix, (w, h))

        for k in results.get('bbox_fields', []):
            if k == 'gt_bboxes_r' or k == 'gt_bboxes_i':
                continue
            warped_bboxes = bt.warp(results[k], matrix, keep_type=True)
            if self.keep_shape:
                iofs = bt.bbox_overlaps(warped_bboxes, img_bound, mode='iof')
                warped_bboxes = warped_bboxes[iofs[:, 0] > self.keep_iof_thr]
            results[k] = warped_bboxes

        for k in results.get('mask_fields', []):
            if k == 'gt_masks_r' or k =='gt_masks_i':
                continue
            polys = switch_mask_type(results[k], 'polygon')
            warped_polys = rotate_polygonmask(polys, matrix, w, h)
            if self.keep_shape:
                obbs = mask2bbox(warped_polys, 'obb')
                iofs = bt.bbox_overlaps(obbs, img_bound, mode='iof')
                index = np.nonzero(iofs[:, 0] > self.keep_iof_thr)[0]
                warped_polys = warped_polys[index]

            if isinstance(results[k], BitmapMasks):
                results[k] = switch_mask_type(warped_polys, 'bitmap')
            elif isinstance(results[k], PolygonMasks):
                results[k] = switch_mask_type(warped_polys, 'polygon')
            else:
                raise NotImplementedError

        for k in results.get('seg_fields', []):
            results[k] = cv2.warpAffine(results[k], matrix, (w, h))

        return results
    def base_rotate(self, results, matrix, w, h, img_bound):
        if 'img_r' in results:
            img_r = cv2.warpAffine(results['img_r'], matrix, (w, h))
            results['img_r'] = img_r
            results['img_shape'] = img_r.shape
        if 'img_i' in results:
            img_i = cv2.warpAffine(results['img_i'], matrix, (w, h))
            results['img_i'] = img_i
            results['img_shape'] = img_i.shape
        if 'gt_masks_r' in results:
            polys = mask2poly(results['gt_masks_r'])
            warped_polys = bt.warp(polys, matrix)
            if self.keep_shape:
                iofs = bt.bbox_overlaps(warped_polys, img_bound, mode='iof')
                if_inwindow = iofs[:, 0] > self.keep_iof_thr
                # if ~if_inwindow.any():
                    # return True
                warped_polys = warped_polys[if_inwindow]
            if isinstance(results['gt_masks_r'], BitmapMasks):
                results['gt_masks_r'] = poly2mask(warped_polys, w, h, 'bitmap')
            elif isinstance(results['gt_masks_r'], PolygonMasks):
                results['gt_masks_r'] = poly2mask(warped_polys, w, h, 'polygon')
            else:
                raise NotImplementedError


            polys = mask2poly(results['gt_masks_i'])
            warped_polys = bt.warp(polys, matrix)
            if self.keep_shape:
                iofs = bt.bbox_overlaps(warped_polys, img_bound, mode='iof')
                if_inwindow = iofs[:, 0] > self.keep_iof_thr
                # if ~if_inwindow.any():
                    # return True
                warped_polys = warped_polys[if_inwindow]
            if isinstance(results['gt_masks_i'], BitmapMasks):
                results['gt_masks_i'] = poly2mask(warped_polys, w, h, 'bitmap')
            elif isinstance(results['gt_masks_i'], PolygonMasks):
                results['gt_masks_i'] = poly2mask(warped_polys, w, h, 'polygon')
            else:
                raise NotImplementedError


            if 'gt_bboxes_r' in results:
                results['gt_bboxes_r'] = bt.bbox2type(warped_polys, 'hbb')
                results['gt_bboxes_i'] = bt.bbox2type(warped_polys, 'hbb')

        elif 'gt_bboxes_r' in results:
            warped_bboxes = bt.warp(results['gt_bboxes_r'], matrix, keep_type=True)
            if self.keep_shape:
                iofs = bt.bbox_overlaps(warped_bboxes, img_bound, mode='iof')
                if_inwindow = iofs[:, 0] > self.keep_iof_thr
                # if ~if_inwindow.any():
                    # return True
                warped_bboxes = warped_bboxes[if_inwindow]
            results['gt_bboxes_r'] = warped_bboxes

            warped_bboxes = bt.warp(results['gt_bboxes_i'], matrix, keep_type=True)
            if self.keep_shape:
                iofs = bt.bbox_overlaps(warped_bboxes, img_bound, mode='iof')
                if_inwindow = iofs[:, 0] > self.keep_iof_thr
                # if ~if_inwindow.any():
                    # return True
                warped_bboxes = warped_bboxes[if_inwindow]
            results['gt_bboxes_i'] = warped_bboxes

        if 'gt_labels_r' in results and self.keep_shape:
            results['gt_labels_r'] = results['gt_labels_r'][if_inwindow]
        if 'gt_labels_i' in results and self.keep_shape:
            results['gt_labels_i'] = results['gt_labels_i'][if_inwindow]
        for k in results.get('aligned_fields', []):
            if self.keep_shape:
                results[k] = results[k][if_inwindow]

        # return False
@PIPELINES.register_module()
class RGBTCollect(Collect):

    def __init__(self,
                 keys,
                 meta_keys=('filename_r', 'filename_i', 'ori_filename', 'ori_shape', 'img_shape',
                            'pad_shape', 'scale_factor', 'h_flip', 'v_flip', 'angle',
                            'matrix', 'rotate_after_flip', 'img_norm_cfg', 'illumination')): #
        super(RGBTCollect, self).__init__(keys, meta_keys)