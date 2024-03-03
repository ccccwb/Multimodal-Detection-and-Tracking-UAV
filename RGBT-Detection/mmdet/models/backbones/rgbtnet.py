import importlib, pdb
import torch
from math import sqrt
from itertools import product as product
import time
import os
import math

import torch.nn as nn

from torch.nn.modules.utils import  _pair
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from torch.nn.modules.activation import PReLU
import torchvision
from mmdet.utils import get_root_logger
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.cnn import (build_conv_layer, build_norm_layer, constant_init,
                      kaiming_init)
from ..builder import BACKBONES
import numpy as np
from ..utils import *
from .resnet import *
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms as transforms
###得到mask
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# mask = cv2.imread('/media/data3/caiwb/RGBTDet/feature_map/mask.jpg',cv2.IMREAD_GRAYSCALE)
# mask = cv2.resize(mask, dsize=[64, 56], interpolation=cv2.INTER_AREA)  
# _, mask = cv2.threshold(mask,200,255,cv2.THRESH_BINARY)
# mask = mask/255.0

# transf = transforms.ToTensor()
# mask_feat3 = transf(mask).unsqueeze(0) # tensor数据格式是torch(C,H,W)  mask_feat3 1 1 56 64
# # print(mask_feat3.shape)
# mask_feat3 = torch.nn.functional.interpolate(mask_feat3, size=[56, 64], scale_factor=None, mode='nearest', align_corners=None)
# plt.imsave ("/media/data3/caiwb/RGBTDet/feature_map/mask_feat3.png", mask, cmap = 'gray')
# mask_feat3 = mask_feat3.to(device = device).float()
##################################

class RGBTBasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None):
        super(RGBTBasicBlock, self).__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'
        #rgb
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp
        
        #红外
        self.norm1_name_t, norm1_t = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name_t, norm2_t = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm1_name_t = self.norm1_name_t + '_t'
        self.norm2_name_t = self.norm2_name_t + '_t'
        self.conv1_t = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name_t, norm1_t)
        self.conv2_t = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name_t, norm2_t)
        self.relu_t = nn.ReLU(inplace=True)

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)
    @property
    def norm1_t(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name_t)

    @property
    def norm2_t(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name_t)

    def forward(self, x1, x2):
        """Forward function"""

        def _inner_forward(x1, x2):
            identity1 = x1

            out1 = self.conv1(x1)
            out1 = self.norm1(out1)
            out1 = self.relu(out1)

            out1 = self.conv2(out1)
            out1 = self.norm2(out1)

            if self.downsample is not None:
                identity1 = self.downsample(x1)

            out1 += identity1

            identity2 = x2

            out2 = self.conv1_t(x2)
            out2 = self.norm1_t(out2)
            out2 = self.relu_t(out2)

            out2 = self.conv2_t(out2)
            out2 = self.norm2_t(out2)

            if self.downsample is not None:
                identity2 = self.downsample(x2)

            out2 += identity2

            return [out1, out2]

        
        out1, out2 = _inner_forward(x1, x2)

        out1 = self.relu(out1)
        out2 = self.relu_t(out2)
        return [out1, out2]

class RGBTBottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None):
        """Bottleneck block for ResNet.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(RGBTBottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        assert plugins is None or isinstance(plugins, list)
        if plugins is not None:
            allowed_position = ['after_conv1', 'after_conv2', 'after_conv3']
            assert all(p['position'] in allowed_position for p in plugins)

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.plugins = plugins
        self.with_plugins = plugins is not None

        if self.with_plugins:
            # collect plugins for conv1/conv2/conv3
            self.after_conv1_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv1'
            ]
            self.after_conv2_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv2'
            ]
            self.after_conv3_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv3'
            ]

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        #rgb
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(norm_cfg, planes * self.expansion, postfix=3)
        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = dcn.pop('fallback_on_stride', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = build_conv_layer(
                conv_cfg,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)
        else:
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            self.conv2 = build_conv_layer(
                dcn,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)
        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        #红外
        self.norm1_name_t, norm1_t = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name_t, norm2_t = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name_t, norm3_t = build_norm_layer(norm_cfg, planes * self.expansion, postfix=3)
        self.norm1_name_t = self.norm1_name_t + '_t'
        self.norm2_name_t = self.norm2_name_t + '_t'
        self.norm3_name_t = self.norm3_name_t + '_t'

        self.conv1_t = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name_t, norm1_t)
        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = dcn.pop('fallback_on_stride', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2_t = build_conv_layer(
                conv_cfg,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)
        else:
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            self.conv2_t = build_conv_layer(
                dcn,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)
        self.add_module(self.norm2_name_t, norm2_t)
        self.conv3_t = build_conv_layer(
            conv_cfg,
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name_t, norm3_t)
        self.relu_t = nn.ReLU(inplace=True)
        self.downsample_t = downsample   




        if self.with_plugins:
            self.after_conv1_plugin_names = self.make_block_plugins(
                planes, self.after_conv1_plugins)
            self.after_conv2_plugin_names = self.make_block_plugins(
                planes, self.after_conv2_plugins)
            self.after_conv3_plugin_names = self.make_block_plugins(
                planes * self.expansion, self.after_conv3_plugins)

    def make_block_plugins(self, in_channels, plugins):
        """ make plugins for block

        Args:
            in_channels (int): Input channels of plugin.
            plugins (list[dict]): List of plugins cfg to build.

        Returns:
            list[str]: List of the names of plugin.

        """
        assert isinstance(plugins, list)
        plugin_names = []
        for plugin in plugins:
            plugin = plugin.copy()
            name, layer = build_plugin_layer(
                plugin,
                in_channels=in_channels,
                postfix=plugin.pop('postfix', ''))
            assert not hasattr(self, name), f'duplicate plugin {name}'
            self.add_module(name, layer)
            plugin_names.append(name)
        return plugin_names

    def forward_plugin(self, x, plugin_names):
        out = x
        for name in plugin_names:
            out = getattr(self, name)(x)
        return out

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name)
    @property
    def norm1_t(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name_t)

    @property
    def norm2_t(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name_t)

    @property
    def norm3_t(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name_t)

    #这里的forward只能有一个输入x，把rgb和红外都放到x里面
    def forward(self, x):
        """Forward function"""
        x1 = x[0]
        x2 = x[1] 
        def _inner_forward(x1, x2):
            identity1 = x1

            out1 = self.conv1(x1)
            out1 = self.norm1(out1)
            out1 = self.relu(out1)

            if self.with_plugins:
                out1 = self.forward_plugin(out1, self.after_conv1_plugin_names)

            out1 = self.conv2(out1)
            out1 = self.norm2(out1)
            out1 = self.relu(out1)

            if self.with_plugins:
                out1 = self.forward_plugin(out1, self.after_conv2_plugin_names)

            out1 = self.conv3(out1)
            out1 = self.norm3(out1)

            if self.with_plugins:
                out1 = self.forward_plugin(out1, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity1 = self.downsample(x1)

            out1 += identity1

            #红外
            identity2 = x2

            out2 = self.conv1_t(x2)
            out2 = self.norm1_t(out2)
            out2 = self.relu_t(out2)

            if self.with_plugins:
                out2 = self.forward_plugin(out2, self.after_conv1_plugin_names)

            out2 = self.conv2_t(out2)
            out2 = self.norm2_t(out2)
            out2 = self.relu_t(out2)

            if self.with_plugins:
                out2 = self.forward_plugin(out2, self.after_conv2_plugin_names)

            out2 = self.conv3_t(out2)
            out2 = self.norm3_t(out2)

            if self.with_plugins:
                out2 = self.forward_plugin(out2, self.after_conv3_plugin_names)

            if self.downsample_t is not None:
                identity2 = self.downsample_t(x2)

            out2 += identity2

            return [out1, out2]


        out1, out2 = _inner_forward(x1, x2)

        out1 = self.relu(out1)
        out2 = self.relu_t(out2)
        return [out1, out2]

class SharedBottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None):
        """Bottleneck block for ResNet.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(SharedBottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        assert plugins is None or isinstance(plugins, list)
        if plugins is not None:
            allowed_position = ['after_conv1', 'after_conv2', 'after_conv3']
            assert all(p['position'] in allowed_position for p in plugins)

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.plugins = plugins
        self.with_plugins = plugins is not None

        if self.with_plugins:
            # collect plugins for conv1/conv2/conv3
            self.after_conv1_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv1'
            ]
            self.after_conv2_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv2'
            ]
            self.after_conv3_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv3'
            ]

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(norm_cfg, planes * self.expansion, postfix=3)
        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = dcn.pop('fallback_on_stride', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = build_conv_layer(
                conv_cfg,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)
        else:
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            self.conv2 = build_conv_layer(
                dcn,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)
        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        if self.with_plugins:
            self.after_conv1_plugin_names = self.make_block_plugins(
                planes, self.after_conv1_plugins)
            self.after_conv2_plugin_names = self.make_block_plugins(
                planes, self.after_conv2_plugins)
            self.after_conv3_plugin_names = self.make_block_plugins(
                planes * self.expansion, self.after_conv3_plugins)

    def make_block_plugins(self, in_channels, plugins):
        """ make plugins for block

        Args:
            in_channels (int): Input channels of plugin.
            plugins (list[dict]): List of plugins cfg to build.

        Returns:
            list[str]: List of the names of plugin.

        """
        assert isinstance(plugins, list)
        plugin_names = []
        for plugin in plugins:
            plugin = plugin.copy()
            name, layer = build_plugin_layer(
                plugin,
                in_channels=in_channels,
                postfix=plugin.pop('postfix', ''))
            assert not hasattr(self, name), f'duplicate plugin {name}'
            self.add_module(name, layer)
            plugin_names.append(name)
        return plugin_names

    def forward_plugin(self, x, plugin_names):
        out = x
        for name in plugin_names:
            out = getattr(self, name)(x)
        return out

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name)

    #这里的forward只能有一个输入x，把rgb和红外都放到x里面
    def forward(self, x):
        """Forward function"""
        x1 = x[0]
        x2 = x[1] 
        def _inner_forward(x1, x2):
            identity1 = x1

            out1 = self.conv1(x1)
            out1 = self.norm1(out1)
            out1 = self.relu(out1)

            if self.with_plugins:
                out1 = self.forward_plugin(out1, self.after_conv1_plugin_names)

            out1 = self.conv2(out1)
            out1 = self.norm2(out1)
            out1 = self.relu(out1)

            if self.with_plugins:
                out1 = self.forward_plugin(out1, self.after_conv2_plugin_names)

            out1 = self.conv3(out1)
            out1 = self.norm3(out1)

            if self.with_plugins:
                out1 = self.forward_plugin(out1, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity1 = self.downsample(x1)

            out1 += identity1

            #红外
            identity2 = x2

            out2 = self.conv1(x2)
            out2 = self.norm1(out2)
            out2 = self.relu(out2)

            if self.with_plugins:
                out2 = self.forward_plugin(out2, self.after_conv1_plugin_names)

            out2 = self.conv2(out2)
            out2 = self.norm2(out2)
            out2 = self.relu(out2)

            if self.with_plugins:
                out2 = self.forward_plugin(out2, self.after_conv2_plugin_names)

            out2 = self.conv3(out2)
            out2 = self.norm3(out2)

            if self.with_plugins:
                out2 = self.forward_plugin(out2, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity2 = self.downsample(x2)

            out2 += identity2

            return [out1, out2]


        out1, out2 = _inner_forward(x1, x2)

        out1 = self.relu(out1)
        out2 = self.relu(out2)
        return [out1, out2]

@BACKBONES.register_module()
class RGBTResNet(nn.Module):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        stem_channels (int): Number of stem channels. Default: 64.
        base_channels (int): Number of base channels of res layer. Default: 64.
        in_channels (int): Number of input image channels. Default: 3.
        num_stages (int): Resnet stages. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the RGBTBottleneck.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        plugins (list[dict]): List of plugins for stages, each dict contains:

            - cfg (dict, required): Cfg dict to build plugin.
            - position (str, required): Position inside block to insert
              plugin, options are 'after_conv1', 'after_conv2', 'after_conv3'.
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity.

    Example:
        >>> from mmdet.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """

    arch_settings = {
        18: (RGBTBasicBlock, (2, 2, 2, 2)),
        34: (RGBTBasicBlock, (3, 4, 6, 3)),
        50: (RGBTBottleneck, (3, 4, 6, 3)),
        101: (RGBTBottleneck, (3, 4, 23, 3)),
        152: (RGBTBottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=64,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 plugins=None,
                 with_cp=False,
                 zero_init_residual=True):
        super(RGBTResNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.depth = depth
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.plugins = plugins
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        self.shared_block = SharedBottleneck
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            if plugins is not None:
                stage_plugins = self.make_stage_plugins(plugins, i)
            else:
                stage_plugins = None
            planes = base_channels * 2**i
            if(i < num_stages/2):
                res_layer = self.make_res_layer(
                    block=self.block,
                    inplanes=self.inplanes,
                    planes=planes,
                    num_blocks=num_blocks,
                    stride=stride,
                    dilation=dilation,
                    style=self.style,
                    avg_down=self.avg_down,
                    with_cp=with_cp,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    dcn=dcn,
                    plugins=stage_plugins)
            else:
                res_layer = self.make_res_layer(
                    block=self.shared_block,
                    inplanes=self.inplanes,
                    planes=planes,
                    num_blocks=num_blocks,
                    stride=stride,
                    dilation=dilation,
                    style=self.style,
                    avg_down=self.avg_down,
                    with_cp=with_cp,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    dcn=dcn,
                    plugins=stage_plugins)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = self.block.expansion * base_channels * 2**(
            len(self.stage_blocks) - 1)

    def make_stage_plugins(self, plugins, stage_idx):
        """ make plugins for ResNet 'stage_idx'th stage .

        Currently we support to insert 'context_block',
        'empirical_attention_block', 'nonlocal_block' into the backbone like
        ResNet/ResNeXt. They could be inserted after conv1/conv2/conv3 of
        RGBTBottleneck.
        An example of plugins format could be:

        >>> plugins=[
        ...     dict(cfg=dict(type='xxx', arg1='xxx'),
        ...          stages=(False, True, True, True),
        ...          position='after_conv2'),
        ...     dict(cfg=dict(type='yyy'),
        ...          stages=(True, True, True, True),
        ...          position='after_conv3'),
        ...     dict(cfg=dict(type='zzz', postfix='1'),
        ...          stages=(True, True, True, True),
        ...          position='after_conv3'),
        ...     dict(cfg=dict(type='zzz', postfix='2'),
        ...          stages=(True, True, True, True),
        ...          position='after_conv3')
        ... ]
        >>> self = ResNet(depth=18)
        >>> stage_plugins = self.make_stage_plugins(plugins, 0)
        >>> assert len(stage_plugins) == 3

        Suppose 'stage_idx=0', the structure of blocks in the stage would be:

        .. code-block:: none

            conv1-> conv2->conv3->yyy->zzz1->zzz2

        Suppose 'stage_idx=1', the structure of blocks in the stage would be:

        .. code-block:: none

            conv1-> conv2->xxx->conv3->yyy->zzz1->zzz2

        If stages is missing, the plugin would be applied to all stages.

        Args:
            plugins (list[dict]): List of plugins cfg to build. The postfix is
                required if multiple same type plugins are inserted.
            stage_idx (int): Index of stage to build

        Returns:
            list[dict]: Plugins for current stage
        """
        stage_plugins = []
        for plugin in plugins:
            plugin = plugin.copy()
            stages = plugin.pop('stages', None)
            assert stages is None or len(stages) == self.num_stages
            # whether to insert plugin into current stage
            if stages is None or stages[stage_idx]:
                stage_plugins.append(plugin)

        return stage_plugins

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``"""
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)
    @property
    def norm1_t(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name_t)
    def _make_stem_layer(self, in_channels, stem_channels):
        #rgb
        self.conv1 = build_conv_layer(
            self.conv_cfg,
            in_channels,
            stem_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False)
        self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, stem_channels, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #红外
        self.conv1_t = build_conv_layer(
            self.conv_cfg,
            in_channels,
            stem_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False)
        self.norm1_name_t, norm1_t = build_norm_layer(self.norm_cfg, stem_channels, postfix=1)
        self.norm1_name_t = self.norm1_name_t + "_t"
        self.add_module(self.norm1_name_t, norm1_t)
        self.relu_t = nn.ReLU(inplace=True)
        self.maxpool_t = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x1 = self.norm1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)
        
        x2 = self.conv1_t(x2)
        x2 = self.norm1_t(x2)
        x2 = self.relu_t(x2)
        x2 = self.maxpool_t(x2)
        outs1 = []
        outs2 = []

        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            [x1, x2] = res_layer([x1, x2])
            if i in self.out_indices:
                outs1.append(x1)
                outs2.append(x2)
        return outs1,outs2

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed"""
        super(RGBTResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
class Cross_attention(nn.Module):
    def __init__(self, channel, N, ratio=1): #channel是通道数 N是特征图大小即HW 
        super(Cross_attention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv1x1_Tk = nn.Conv2d(channel*2, channel, kernel_size=1).cuda() 
        self.conv1x1_Tv = nn.Conv2d(channel*2, channel, kernel_size=1).cuda() 

        self.query = torch.nn.Parameter(torch.FloatTensor(1, channel, N).cuda(), requires_grad=True).data.fill_(1e-23) # 可学习的query，初始化为0

    def forward(self, x_v, x_i):
        B, C, W, H = x_v.size()

        rgbt_feats = torch.cat((x_v,x_i), dim=1)

        Tk_feats = self.conv1x1_Tk(rgbt_feats).view(B,C,-1)
        Tv_feats = self.conv1x1_Tv(rgbt_feats).view(B,C,-1)
        list = []
        for b in range(0,B):
            energy_res = torch.bmm(self.query.permute(0, 2, 1), Tk_feats[b].unsqueeze(0)) #wh, wh44
            cross_attention_res = torch.nn.functional.softmax(energy_res,dim=-1)
            list.append(cross_attention_res)
        cross_attention = torch.cat([cross_attention_res for cross_attention_res in list], dim = 0)
        # energy = torch.bmm(self.query.permute(0,2,1), Tk_feats) #wh, wh
        # cross_attention = torch.nn.functional.softmax(energy,dim=-1)
        out = torch.bmm(Tv_feats, cross_attention.permute(0, 2, 1)) 
        out = out.view(B, C, W, H)

        return out
        

@BACKBONES.register_module()
class RgbtNet(nn.Module):

    def __init__(self):

        super(RgbtNet, self).__init__()
        # self.n_classes = n_classes

        self.model = RGBTResNet(depth=50, num_stages=4, out_indices=(0, 1, 2, 3), frozen_stages=1, norm_cfg=dict(type='BN', requires_grad=True), norm_eval=True, style='pytorch')
        self.feat_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.feat_1_bn = nn.BatchNorm2d(256, momentum=0.01)
        self.feat_2 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.feat_2_bn = nn.BatchNorm2d(512, momentum=0.01)
        self.feat_3 = nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.feat_3_bn = nn.BatchNorm2d(1024, momentum=0.01)
        self.feat_4 = nn.Conv2d(4096, 2048, kernel_size=3, stride=1, padding=1, bias=False)
        self.feat_4_bn = nn.BatchNorm2d(2048, momentum=0.01)

        self.conv_att3 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1, bias=False)


        # self.cross_attention_3 = Cross_attention(channel = 1024, N = 56*64)
        # self.cross_attention_4 = Cross_attention(channel = 2048, N = 28*32)

    def abc(self, x, pre=None, mask=None):
        """
        inputs :
            x : input feature maps( B X C X W X H)
        returns :
            out : self attention value + input feature
            attention: B X N X N (N is Width*Height)
        """
        B, C, W, H = x.size()
        proj_query = self.query_conv(x).view(B, -1, W * H)  # B X (N)X C
        proj_key = proj_query  # B X C x (N)

        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)  # transpose check
        self.softmax = nn.Softmax(dim=-1)
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = x.view(B, -1, W * H)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, W, H)

        out = self.gamma * out + x

        if type(pre) != type(None):
            # using long distance attention layer to copy information from valid regions
            context_flow = torch.bmm(pre.view(B, -1, W*H), attention.permute(0, 2, 1)).view(B, -1, W, H)
            context_flow = self.alpha * (1-mask) * context_flow + (mask) * pre
            out = self.model(torch.cat([out, context_flow], dim=1))

        return out, attention
    
    def visualize_feature_map(self, img_batch):
        C, W, H = img_batch.size() # b为batch

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
        plt.savefig('/media/data3/caiwb/RGBTDet/feature_map/feature_map.png', camp = 'rainbow')
        plt.show()
        plt.figure()
        # 各个特征图按1：1 叠加
        feature_map_sum = sum(ele for ele in feature_map_combination)
        # print("feature_map_sum",feature_map_sum.shape)
        # print(feature_map_sum)

        
        feature_map_sum = (feature_map_sum - feature_map_sum.min()) / (feature_map_sum.max() - feature_map_sum.min()) #* mask_feat3[0][0].cpu()
        
        plt.imshow(feature_map_sum)

        plt.imsave ("/media/data3/caiwb/RGBTDet/feature_map/feature_map_sum.png", feature_map_sum.cpu(), cmap = 'rainbow')#rainbow
    def cross_attention_3(self, vis, lr):
        B, C, W, H = vis.size() # b为batch
        prj_key = prj_value = self.conv_att3(vis).view(B, -1, W * H) # b c w*h
        prj_query = self.conv_att3(lr).view(B, -1, W * H) # b c w*h

        energy = torch.bmm(prj_query.permute(0,2,1), prj_key) #wh, wh
        cross_attention = torch.nn.functional.softmax(energy,dim=-1)
        out = torch.bmm(prj_value, cross_attention.permute(0, 2, 1)) 
        out = out.view(B, C, W, H)# * mask_feat3.repeat(B, C, 1, 1)

        out = vis + out
        # feature = (out + lr) 
        # feature = feature.reshape(feature.shape[1:])
        # self.visualize_feature_map(feature)
        return out


    def forward(self, image_vis, image_lwir):
        """
        Forward propagation.

        :param image: images, a tensor of dimensions (N, 3, 300, 300)
        :return: 8732 locations and class scores (i.e. w.r.t each prior box) for each image
        """
        # Run VGG base network convolutions (lower level feature map generators)
        [conv1vis, conv2vis , conv3vis, conv4vis], [conv1ir, conv2ir , conv3ir, conv4ir] = self.model(image_vis, image_lwir)
        conv1_feats = torch.cat([conv1vis, conv1ir], dim=1)
        conv1_feats = F.relu(self.feat_1_bn(self.feat_1(conv1_feats)))

        conv2_feats = torch.cat([conv2vis, conv2ir], dim=1)
        conv2_feats = F.relu(self.feat_2_bn(self.feat_2(conv2_feats)))

        conv3vis = self.cross_attention_3(conv3vis,conv3ir)
        conv3_feats = torch.cat([conv3vis, conv3ir], dim = 1)
        conv3_feats = F.relu(self.feat_3_bn(self.feat_3(conv3_feats)))

        conv4_feats = torch.cat([conv4vis, conv4ir], dim = 1)
        conv4_feats = F.relu(self.feat_4_bn(self.feat_4(conv4_feats)))
        # conv3_feats = self.cross_attention_3(conv3vis,conv3ir)
        # conv4_feats = self.cross_attention_4(conv4vis,conv4ir)

        return (conv1_feats, conv2_feats, conv3_feats, conv4_feats)
        
    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        t_start = time.time()
        if isinstance(pretrained, str):
            raw_state_dict = torch.load(pretrained)
            if 'model' in raw_state_dict.keys():
                raw_state_dict = raw_state_dict['model']
            state_dict = {}
            for k, v in raw_state_dict.items():

                if k.find('conv1') >= 0:
                    state_dict[k] = v
                    if(k.find('layer3') >= 0 or k.find('layer4')>= 0):#3,4为共享层
                        continue
                    state_dict[k.replace('conv1', 'conv1_t')] = v
                if k.find('conv2') >= 0:
                    state_dict[k] = v
                    if(k.find('layer3')>= 0 or k.find('layer4')>= 0):#3,4为共享层
                        continue
                    state_dict[k.replace('conv2', 'conv2_t')] = v
                if k.find('conv3') >= 0:
                    state_dict[k] = v
                    if(k.find('layer3')>= 0 or k.find('layer4')>= 0):#3,4为共享层
                        continue
                    state_dict[k.replace('conv3', 'conv3_t')] = v 
                if k.find('bn1') >= 0:
                    state_dict[k] = v
                    if(k.find('layer3')>= 0 or k.find('layer4')>= 0):#3,4为共享层
                        continue
                    state_dict[k.replace('bn1', 'bn1_t')] = v 
                if k.find('bn2') >= 0:
                    state_dict[k] = v
                    if(k.find('layer3')>= 0 or k.find('layer4')>= 0):#3,4为共享层
                        continue
                    state_dict[k.replace('bn2', 'bn2_t')] = v 
                if k.find('bn3') >= 0:
                    state_dict[k] = v
                    if(k.find('layer3')>= 0 or k.find('layer4')>= 0):#3,4为共享层
                        continue
                    state_dict[k.replace('bn3', 'bn3_t')] = v 
                    
                if k.find('downsample') >= 0:
                    state_dict[k] = v
                    if(k.find('layer3')>= 0 or k.find('layer4')>= 0):#3,4为共享层
                        continue
                    state_dict[k.replace('downsample', 'downsample_t')] = v

            self.model.load_state_dict(state_dict, strict=True)
            del state_dict
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

        

        t_end = time.time()
        # logger.info(
        # "Load model, Time usage:\n\tIO:, initialize parameters: {}".format(
        #     t_end - t_end))