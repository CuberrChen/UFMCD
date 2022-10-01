#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/12/17 下午9:24
# @Author  : chenxb
# @FileName: config.py
# @Software: PyCharm
# @mail ：joyful_chen@163.com

from easydict import EasyDict

config = EasyDict()

config.NUM_CLASSES = 21 # 21 for PASCAL-VOC
config.MODEL_NAME = 'DeepLabv2'
config.IGNORE_LABEL = 255
config.RAMPUP_LENGTH = 2500
config.UFMCD = EasyDict()
config.UFMCD.SUP_WEIGHT = 1.0
config.UFMCD.CPS_WEIGHT = 1.0
config.UFMCD.RAMPUP_LENGTH = config.RAMPUP_LENGTH
config.UFMCD.USE_CUTMIX = True
config.UFMCD.MASK_PROP_RANGE = (0.25, 0.5)
config.UFMCD.BOXMASK_N_BOXES = 3
config.UFMCD.BOXMASK_FIXED_ASPECT_RATIO = False
config.UFMCD.BOXMASK_BY_SIZE = False
config.UFMCD.BOXMASK_OUTSIDE_BOUNDS = False
config.UFMCD.BOXMASK_NO_INVERT = False