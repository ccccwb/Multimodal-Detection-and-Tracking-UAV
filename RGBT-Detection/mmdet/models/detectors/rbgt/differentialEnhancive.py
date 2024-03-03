import torch
import torch.nn.functional as F
from torch import nn, Tensor
import mmcv
from mmcv.cnn import (build_conv_layer, build_norm_layer)
class RGBTdiffEnhancive(nn.Module):
    def __init__(self, d_model = 512, scale = 16):
        super().__init__()
        self.fc1=nn.Sequential(nn.Conv2d(d_model, d_model//scale, kernel_size=1, stride=1, bias=False),
                               nn.BatchNorm2d(d_model//scale, momentum=0.01),
                               nn.ReLU(inplace=True))
        self.fc2=nn.Conv2d(d_model//scale, d_model*2, kernel_size=1, stride=1, bias=False) 
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=1) #指定维度1

        
    def forward(self, feat_rgb, feat_ir):
        b, c, h, w = feat_rgb.shape
        feat_diff = feat_rgb - feat_ir

        feat_diff = self.gap(feat_diff)

        feat_diff = self.fc1(feat_diff)  # 降维

        rgb_ir = self.fc2(feat_diff)  # 升维，升到两倍c

        rgb_ir = rgb_ir.reshape(b, 2, c, -1) #调整形状，变为 两个全连接层的值
        rgb_ir = self.softmax(rgb_ir)

        rgb_specific, ir_specific = rgb_ir.chunk(2, dim=1) #两个模态分开

        rgb_specific = rgb_specific.reshape(b, c, 1, 1)*feat_rgb

        ir_specific = ir_specific.reshape(b, c, 1, 1)*feat_ir
        out = [rgb_specific, ir_specific]
        return out
