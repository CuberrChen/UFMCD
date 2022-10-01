# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager
from paddleseg.models import layers
from paddleseg.utils import utils

__all__ = ['DeepLabV2']

@manager.MODELS.add_component
class DeepLabV2(nn.Layer):
    """
    The DeepLabV2 implementation based on PaddlePaddle.
    """
    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices=(3, ),
                 aspp_ratios=(6, 12, 18, 24),
                 align_corners=False,
                 pretrained=None):
        super().__init__()

        self.backbone = backbone
        backbone_channels = [
            backbone.feat_channels[i] for i in backbone_indices
        ]

        self.head = DeepLabV2Head(num_classes, backbone_indices,
                                  backbone_channels, aspp_ratios, align_corners)
        self.align_corners = align_corners
        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        feat_list = self.backbone(x)
        logit_list = self.head(feat_list)
        ori_shape = paddle.shape(x)[2:]
        return [F.interpolate(logit_list, ori_shape, mode='bilinear', align_corners=self.align_corners)]

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


class ASPPModule(nn.Layer):
    """
    Atrous Spatial Pyramid Pooling.
    Args:
        aspp_ratios (tuple): The dilation rate using in ASSP module.
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.
    """
    def __init__(self,
                 aspp_ratios,
                 in_channels,
                 out_channels,
                 align_corners,
                 data_format='NCHW'):
        super().__init__()

        self.align_corners = align_corners
        self.data_format = data_format
        self.aspp_blocks = nn.LayerList()

        for ratio in aspp_ratios:
            block = nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                dilation=ratio,
                padding=0 if ratio == 1 else ratio,
                data_format=data_format)
            self.aspp_blocks.append(block)
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                m.bias.set_value(np.zeros(m.bias.shape).astype('float32'))

    def forward(self, x):
        return sum([block(x) for block in self.aspp_blocks])

class DeepLabV2Head(nn.Layer):
    """
    The DeepLabV2Head implementation based on PaddlePaddle.
    Args:
        Please Refer to DeepLabV2PHead above.
    """

    def __init__(self, num_classes, backbone_indices, backbone_channels,
                 aspp_ratios, align_corners):
        super().__init__()

        self.aspp = ASPPModule(
            aspp_ratios,
            backbone_channels[0],
            num_classes,
            align_corners)
        self.backbone_indices = backbone_indices

    def forward(self, feat_list):
        x = feat_list[self.backbone_indices[0]]
        x = self.aspp(x)
        return x
