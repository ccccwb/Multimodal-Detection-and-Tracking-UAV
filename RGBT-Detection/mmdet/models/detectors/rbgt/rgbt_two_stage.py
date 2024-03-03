import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

# from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck

from mmdet.core import auto_fp16
from mmdet.models.losses.smooth_l1_loss import smooth_l1_loss
import matplotlib.pyplot as plt
import cv2
import mmcv
import numpy as np
import BboxToolkit as bt
import warnings
from collections import OrderedDict

from ..obb.obb_base import OBBBaseDetector
from ..obb.obb_test_mixins import RotateAugRPNTestMixin

from .rgbt_deformable_attention import RgbtDeformableTransformerAttention
from .rgbt_deformable_encoder import RGBTDeformableTransformer
from .position_encoding import Joiner, build_position_encoding

@DETECTORS.register_module()
class RGBTTwoStageDetector(OBBBaseDetector, RotateAugRPNTestMixin):
    """Base class for two-stage detectors.
    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """
    def __init__(self,
                 backbone_r,
                 backbone_i,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RGBTTwoStageDetector, self).__init__()
        self.backbone_r = build_backbone(backbone_r)
        self.backbone_i = build_backbone(backbone_i) # 23.28M

        self.position_encoding = 'sine'


        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_fusion_layer()
        self.init_weights(pretrained=pretrained)


    def init_fusion_layer(self):
        ''' 初始化融合层， 少数自己写的代码 '''
        num_level = 4 # 特征的尺度，对应resnet四个阶段，后续得想办法直接对应起来
        d_models = [256, 512, 1024, 2048] # 四个尺度特征的通道数
        d_model = 512

        ''' conv1x1 降维卷积层  可以整合进fpn, 先看看有没有效果'''
        self.conv1x1_r = nn.ModuleList(
            nn.Conv2d(d_models[lvl], d_model, kernel_size=1, stride=1, bias=False) for lvl in range(num_level)
        )
        self.conv1x1_i = nn.ModuleList(
            nn.Conv2d(d_models[lvl], d_model, kernel_size=1, stride=1, bias=False) for lvl in range(num_level)
        )

        #判断有没有位置编码 感觉后面可以去掉这个判断，默认有
        if self.position_encoding is not None:
            ''' 为每个尺度构建一个位置编码层，相同尺度下的可见光和红外特征共享'''
            self.position_encoding_lvl = nn.ModuleList(build_position_encoding(self.position_encoding, d_model) for lvl in range(num_level)) 
        ''' 为每个尺度的特征构建一个tansformer实际上只有encoder部分  
        其中num_feature_levels实际上是模态数 后面要改，
        dim_feedforward也得改下  得和d_model对应上'''
        
        self.RGBTTransformer = nn.ModuleList(# 224.467M  其中 LVL3 168.649m
            RGBTDeformableTransformer(
                d_model=d_model, nhead=4, num_encoder_layers=4, dim_feedforward=d_model*4, dropout=0.1, activation="relu", num_feature_levels=2, enc_n_points=4
            ) 
            for lvl in range(num_level)
        )

        ''' 将 Tansformer 输出的特征合并 '''
        self.fusion_feat = nn.ModuleList( #149.82m
            nn.Sequential(
                nn.Conv2d(d_model*2, d_model, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(d_model, momentum=0.01),
                nn.ReLU(),
                nn.Conv2d(d_model, d_model, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(d_model, momentum=0.01)
            )
            for lvl in range(num_level)
        )
        ''' 将原始可见光和红外特征进行合并  作为残差 '''
        self.res_feat = nn.ModuleList( #149.82m
            nn.Sequential(
                nn.Conv2d(d_model*2, d_model, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(d_model, momentum=0.01),
                nn.ReLU(),
                nn.Conv2d(d_model, d_model, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(d_model, momentum=0.01)
            )
            for lvl in range(num_level)
        )
        self.relu = nn.ReLU(inplace=True)


    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(RGBTTwoStageDetector, self).init_weights(pretrained)
        self.backbone_r.init_weights(pretrained=pretrained)
        self.backbone_i.init_weights(pretrained=pretrained)

        print("init_weights")
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_roi_head:
            self.roi_head.init_weights(pretrained)
    def visualize_feature_map(self, img_batch, lvl):
        C, W, H = img_batch.size() 

        feature_map = img_batch
        print("feature_map",feature_map.shape)
    
        feature_map_combination = []
        plt.figure()
    
        num_pic = feature_map.shape[0]
        squr = num_pic ** 0.5
        row = round(squr)
        col = row + 1 if squr - row > 0 else row
        # print(row,col)
        feature_map = feature_map.cpu()
        plt.get_cmap('gray')
        for i in range(0, num_pic):
            feature_map_split = feature_map[i, :, :]
            # print(feature_map_split.shape)
            feature_map_combination.append(feature_map_split)
            plt.subplot(row, col, i + 1)
            plt.imshow(feature_map_split)
            plt.axis('off')
            plt.imsave ("/media/data3/caiwb/RGBTDetection/feature_map/feature_map"+str(lvl) + '_' + str(i) + '.png', feature_map_split.cpu(), cmap = 'jet')#rainbow

        plt.savefig('/media/data3/caiwb/RGBTDetection/feature_map/feature_map'+str(lvl)+'.png', camp = 'jet')
        plt.show()
        plt.figure()
        # 各个特征图按1：1 叠加
        feature_map_sum = sum(ele for ele in feature_map_combination)
        # print("feature_map_sum",feature_map_sum.shape)
        # print(feature_map_sum)
                
        feature_map_sum = (feature_map_sum - feature_map_sum.min()) / (feature_map_sum.max() - feature_map_sum.min()) #* mask_feat3[0][0].cpu()
        
        plt.imshow(feature_map_sum)

        plt.imsave ("/media/data3/caiwb/RGBTDetection/feature_map/feature_map_sum"+str(lvl)+'.png', feature_map_sum.cpu(), cmap = 'jet')#rainbow


    def extract_feat_r(self, img_r):
        """Directly extract features from the backbone+neck
        """
        x_r  = self.backbone_r(img_r)
        return x_r 

    def extract_feat_i(self, img_i):
        """Directly extract features from the backbone+neck
        """
        x_i  = self.backbone_i(img_i)
        return x_i 

    def extract_feat(self, img_r, img_i):
        """Directly extract features from the backbone+neck
        """
        if self.position_encoding != None:
            x_rs = self.extract_feat_r(img_r)
            x_is = self.extract_feat_i(img_i)
            x_f = []
            x_r_banlance = []
            x_i_banlance = []
            for lvl, (x_r, x_i) in enumerate(zip(x_rs, x_is)):
                x_r = self.conv1x1_r[lvl](x_r)
                x_i = self.conv1x1_i[lvl](x_i)
                x_r_banlance.append(x_r)
                x_i_banlance.append(x_i)
                b, c, h, w = x_r.shape

                x_input = [x_r, x_i]
                #位置编码
                pos_input = [self.position_encoding_lvl[lvl](x_r), self.position_encoding_lvl[lvl](x_i)]
                ''' deformable transformer 融合的特征与原始特征做残差 '''
                # deformable transformer 融合的特征
                memory_fusion = self.fusion_feat[lvl](self.RGBTTransformer[lvl](x_input, pos_input).transpose(1, 2).contiguous().view((b, 2*c, h, w))) 
                
                # 两个模态特征直接卷积
                res_feat = self.res_feat[lvl](torch.cat(x_input, dim=1))
                
                x_f.append(self.relu(memory_fusion + res_feat)) 
                # if lvl != 0:
                #     continue
                # self.visualize_feature_map(x_f[lvl].reshape(x_f[lvl].shape[1:]), lvl)

            x_f = tuple(x_f)
            if self.with_neck:
                x_f = self.neck(x_f)
            if self.training and self.with_neck:    
                with torch.no_grad():
                      x_r_banlance = tuple(x_r_banlance)
                      x_i_banlance = tuple(x_i_banlance)
                      x_r_banlance = self.neck(x_r_banlance)
                      x_i_banlance = self.neck(x_i_banlance)
                return (x_f, x_r_banlance, x_i_banlance)     
            return x_f 

    def forward_dummy(self, img_r):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        outs = ()
        # backbone
        x  = self.extract_feat(img_r, img_r)
        # rpn
        proposal_type = 'hbb'
        if self.with_rpn:
            proposal_type = getattr(self.rpn_head, 'bbox_type', 'hbb')
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )

        if proposal_type == 'hbb':
            proposals = torch.randn(1000, 4).to(img_r.device)
        elif proposal_type == 'obb':
            proposals = torch.randn(1000, 5).to(img_r.device)
        else:
            # poly proposals need to be generated in roi_head
            proposals = None
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def forward_train(self,
                      img_r,
                      img_metas,
                      img_i,
                      gt_bboxes_r,
                      gt_obboxes_r,
                      gt_labels_r,
                      gt_bboxes_i,
                      gt_obboxes_i,
                      gt_labels_i,
                      gt_bboxes_ignore=None,
                      gt_obboxes_ignore=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x, x_r, x_i   = self.extract_feat(img_r, img_i)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_type = getattr(self.rpn_head, 'bbox_type', 'hbb')
            target_bboxes = gt_bboxes_i if proposal_type == 'hbb' else gt_obboxes_i
            target_bboxes_ignore = gt_bboxes_ignore if proposal_type == 'hbb' \
                    else gt_obboxes_ignore
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                target_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=target_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals
        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes_i, gt_obboxes_i, gt_labels_i,
                                                 gt_bboxes_ignore, gt_obboxes_ignore,
                                                 **kwargs)
        losses.update(roi_losses)
        with torch.no_grad():
            rgb_losses = self.roi_head.forward_train(x_r, img_metas, proposal_list,
                                                 gt_bboxes_i, gt_obboxes_i, gt_labels_i,
                                                 gt_bboxes_ignore, gt_obboxes_ignore,
                                                 **kwargs)
            ir_losses = self.roi_head.forward_train(x_i, img_metas, proposal_list,
                                                    gt_bboxes_i, gt_obboxes_i, gt_labels_i,
                                                    gt_bboxes_ignore, gt_obboxes_ignore,
                                                    **kwargs)
            coeff = {"rgb_cls":rgb_losses['loss_cls'], "ir_cls": ir_losses['loss_cls'], "rgb_bbox":rgb_losses['loss_bbox'], "ir_bbox":ir_losses['loss_bbox']}
            # print(coeff)
            losses.update(coeff)
        return losses

    async def async_simple_test(self,
                                img_r,
                                img_meta,
                                img_i,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x  = self.extract_feat(img_r, img_i)
        
        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img_r, img_metas, img_i, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x  = self.extract_feat(img_r, img_i)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, img_r, img_metas, img_i, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        x = self.extract_feats(img_r, img_i)
        proposal_list = self.rotate_aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)

    @auto_fp16(apply_to=('img_r','img_i' ))
    def forward(self, img_r, img_metas, img_i, return_loss=True, **kwargs):
        """
        Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=True`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        self.training = return_loss
        if return_loss:
            return self.forward_train(img_r, img_metas, img_i, **kwargs)
        else:
            return self.forward_test(img_r, img_metas, img_i, **kwargs)

    def forward_test(self, img_r, img_metas, img_i, **kwargs):
            """
            Args:
                imgs (List[Tensor]): the outer list indicates test-time
                    augmentations and inner Tensor should have a shape NxCxHxW,
                    which contains all images in the batch.
                img_metas (List[List[dict]]): the outer list indicates test-time
                    augs (multiscale, flip, etc.) and the inner list indicates
                    images in a batch.
            """
            for var, name in [(img_r, 'imgs'), (img_metas, 'img_metas')]:
                if not isinstance(var, list):
                    raise TypeError(f'{name} must be a list, but got {type(var)}')

            num_augs = len(img_r)
            if num_augs != len(img_metas):
                raise ValueError(f'num of augmentations ({len(img_r)}) '
                                f'!= num of image meta ({len(img_metas)})')
                 # TODO: remove the restriction of samples_per_gpu == 1 when prepared
            samples_per_gpu = img_r[0].size(0)
            assert samples_per_gpu == 1

            if num_augs == 1:
                """
                proposals (List[List[Tensor]]): the outer list indicates test-time
                    augs (multiscale, flip, etc.) and the inner list indicates
                    images in a batch. The Tensor should have a shape Px4, where
                    P is the number of proposals.
                """
                if 'proposals' in kwargs:
                    kwargs['proposals'] = kwargs['proposals'][0]
                return self.simple_test(img_r[0], img_metas[0], img_i[0], **kwargs)
            else:
                # TODO: support test augmentation for predefined proposals
                assert 'proposals' not in kwargs
                return self.aug_test(img_r, img_metas, **kwargs)
    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    colors='green',
                    thickness=1.,
                    font_size=10,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None,
                    original_img=''):

        self.CLASSES = ('car', 'bus', 'truck', 'van', 'freight_car')   
        if original_img != '':
            original_img = mmcv.imread(original_img)
            original_img = original_img.copy()
        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ] 
        labels = np.concatenate(labels)
        # draw segmentation masks
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            inds = np.where(bboxes[:, -1] > score_thr)[0]
            np.random.seed(42)
            color_masks = [
                np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                for _ in range(max(labels) + 1)
            ]
            for i in inds:
                i = int(i)
                color_mask = color_masks[labels[i]]
                mask = segms[i]
                img[mask] = img[mask] * 0.5 + color_mask * 0.5
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        bboxes, scores = bboxes[:, :-1], bboxes[:, -1]
        if original_img != '':
            img = bt.imshow_bboxes(
                original_img,
                bboxes,
                labels,
                scores=scores,
                class_names=self.CLASSES,
                score_thr=score_thr,
                colors=colors,#['green', 'blue', 'red', 'yellow', 'orange']
                thickness=thickness,
                font_size=font_size,
                win_name=win_name,
                show=show,
                wait_time=wait_time,
                out_file=out_file)
        else:
            img = bt.imshow_bboxes(
                img,
                bboxes,
                labels,
                scores=scores,
                class_names=self.CLASSES,
                score_thr=score_thr,
                colors=colors,#['green', 'blue', 'red', 'yellow', 'orange'],
                thickness=thickness,
                font_size=font_size,
                win_name=win_name,
                show=show,
                wait_time=wait_time,
                out_file=out_file)


        if not (show or out_file):
            warnings.warn('show==False and out_file is not specified, only '
                          'result image will be returned')
            return img

