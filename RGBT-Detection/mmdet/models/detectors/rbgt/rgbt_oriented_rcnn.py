from mmdet.models.builder import DETECTORS
from .rgbt_two_stage import RGBTTwoStageDetector

@DETECTORS.register_module()
class RGBTOrientedRCNN(RGBTTwoStageDetector):

    def __init__(self,
                 backbone_r,
                 backbone_i,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RGBTOrientedRCNN, self).__init__(
            backbone_r = backbone_r,
            backbone_i = backbone_i,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
