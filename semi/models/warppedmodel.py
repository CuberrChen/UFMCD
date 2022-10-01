#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/1/13 下午6:25
# @Author  : chenxb
# @FileName: warppedmodel.py
# @Software: PyCharm
# @mail ：joyful_chen@163.com

import os
import copy
import paddle
import paddle.nn as nn
from paddleseg.utils import logger


class WrappedModel(nn.Layer):
    def __init__(self, model_l):
        super(WrappedModel, self).__init__()
        self.model_l = model_l
        self.model_r = copy.deepcopy(model_l)

    def forward(self, images, step=1, feature=None, lam=None):
        if not self.training or step==1:
            return self.model_l(images)
        if step == 2:
            return self.model_r(images)
        if step == 3:
            feat_list = self.model_l.backbone(images)
            logit_list = self.model_l.head(feat_list)
            ori_shape = paddle.shape(images)[2:]
            feature = feat_list[self.model_l.head.backbone_indices[0]]
            return [nn.functional.interpolate(logit_list, ori_shape, mode='bilinear', align_corners=self.model_l.align_corners)] , feature
        if step == 4:
            feat_list = self.model_r.backbone(images)
            logit_list = self.model_r.head(feat_list)
            ori_shape = paddle.shape(images)[2:]
            feature = feat_list[self.model_r.head.backbone_indices[0]]
            return [nn.functional.interpolate(logit_list, ori_shape, mode='bilinear', align_corners=self.model_r.align_corners)] , feature
        if step == 5: # feature mix: label with unlabel lmodel
            feat_list = self.model_l.backbone(images)
            feat_list[self.model_l.head.backbone_indices[0]] = lam * feature + (1 - lam) * feat_list[self.model_l.head.backbone_indices[0]]
            logit_list = self.model_l.head(feat_list)
            ori_shape = paddle.shape(images)[2:]
            return [nn.functional.interpolate(logit_list, ori_shape, mode='bilinear', align_corners=self.model_l.align_corners)]
        if step == 6: # feature mix: label with unlabel lmodel
            feat_list = self.model_r.backbone(images)
            feat_list[self.model_r.head.backbone_indices[0]] = lam * feature + (1 - lam) * feat_list[self.model_r.head.backbone_indices[0]]
            logit_list = self.model_r.head(feat_list)
            ori_shape = paddle.shape(images)[2:]
            return [nn.functional.interpolate(logit_list, ori_shape, mode='bilinear', align_corners=self.model_r.align_corners)]


    def resume(self, optimizer, optimizer_r, resume_model):
        if resume_model is not None:
            logger.info('Resume model from {}'.format(resume_model))
            if os.path.exists(resume_model):
                resume_model = os.path.normpath(resume_model)
                ckpt_path = os.path.join(resume_model, 'model.pdparams')
                para_state_dict = paddle.load(ckpt_path)
                ckpt_path = os.path.join(resume_model, 'model.pdopt')
                opti_state_dict = paddle.load(ckpt_path)
                optimizer_dict = opti_state_dict['optim_l']
                optimizer_r_dict = opti_state_dict['optim_r']
                self.set_state_dict(para_state_dict)
                optimizer.set_state_dict(optimizer_dict)
                optimizer_r.set_state_dict(optimizer_r_dict)
                iter = resume_model.split('_')[-1]
                iter = int(iter)
                return iter
            else:
                raise ValueError(
                    'Directory of the model needed to resume is not Found: {}'.
                        format(resume_model))
        else:
            logger.info('No model needed to resume.')