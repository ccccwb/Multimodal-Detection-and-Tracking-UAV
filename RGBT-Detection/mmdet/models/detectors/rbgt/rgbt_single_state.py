import torch.nn as nn

from mmdet.core import arb2result
from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck
from ..obb.obb_base import OBBBaseDetector

from mmdet.core import auto_fp16

@DETECTORS.register_module()
class RGBTSingleStageDetector(OBBBaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RGBTSingleStageDetector, self).__init__()
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(RGBTSingleStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img, img_t):
        """Directly extract features from the backbone+neck
        """
        x  = self.backbone(img, img_t)
        if self.with_neck:
            x = self.neck(x)
        return x 

    def forward_dummy(self, img, img_t):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        x  = self.extract_feat(img, img_t)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      img_t,
                      gt_bboxes,
                      gt_obboxes,
                      gt_labels,
                      gt_bboxes_t,
                      gt_obboxes_t,
                      gt_labels_t,
                      gt_bboxes_ignore=None,
                      gt_obboxes_ignore=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        x  = self.extract_feat(img, img_t)
        losses = self.bbox_head.forward_train(x, img_metas, gt_obboxes,
                                              gt_labels, gt_obboxes_ignore)
        return losses
    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, img_t, return_loss=True, **kwargs):
        """
        Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=True`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(img, img_metas, img_t, **kwargs)
        else:
            return self.forward_test(img, img_metas, img_t, **kwargs)

    def simple_test(self, img, img_metas, img_t, rescale=False):
        """Test function without test time augmentation

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            np.ndarray: proposals
        """
        x = self.extract_feat(img, img_t)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        bbox_type = getattr(self.bbox_head, 'bbox_type', 'hbb')
        bbox_results = [
            arb2result(det_bboxes, det_labels, self.bbox_head.num_classes, bbox_type)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results[0]

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation"""
        raise NotImplementedError
    def forward_test(self, imgs, img_metas, img_t, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')
        # TODO: remove the restriction of samples_per_gpu == 1 when prepared
        samples_per_gpu = imgs[0].size(0)
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
            return self.simple_test(imgs[0], img_metas[0], img_t[0], **kwargs)
        else:
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(imgs, img_metas, **kwargs)
