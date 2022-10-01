#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/12/31 下午7:58
# @Author  : chenxb
# @FileName: augmentation.py
# @Software: PyCharm
# @mail ：joyful_chen@163.com

import copy
import math
import numpy as np
import scipy.stats
from scipy.ndimage import gaussian_filter

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class Collator(object):
    def __init__(self, cutmix=False,
                cutmix_beta=1.0):
        super(Collator, self).__init__()
        self.cutmix = CutMix(beta=cutmix_beta) if cutmix else None

    def __call__(self, batch_data):
        for data in batch_data:
            pass



class CutMix:
    """
    Props to https://github.com/clovaai/CutMix-Pypaddle/blob/master/train.py#L228
    """
    def __init__(self,beta=1.0):
        self.beta = beta

    def get_rand_bbox(self, size, lam):

        # Get cutout size
        H = size[2]
        W = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # Sample location uniformly at random
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        # Clip
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2


    def cutmix(self, images_1, images_2, labels_1=None, labels_2=None, beta=1.0):

        # Determine randomly which is the patch
        if np.random.rand() > 0.5:
            images_1, images_2 = images_2, images_1
            labels_1, labels_2 = labels_2, labels_1

        # Randomly sample lambda from beta distribution
        lam = np.random.beta(beta, beta)

        # Get bounding box
        bbx1, bby1, bbx2, bby2 = self.get_rand_bbox(images_1.shape, lam)

        # Cut and paste images and labels
        images, labels = copy.copy(images_1), copy.copy(labels_1)
        images[:, :, bbx1:bbx2, bby1:bby2] = images_2[:, :, bbx1:bbx2, bby1:bby2]
        labels[:, :, bbx1:bbx2, bby1:bby2] = labels_2[:, :, bbx1:bbx2, bby1:bby2]
        return images, labels

    @paddle.no_grad()
    def __call__(self, images_1, images_2, labels_1, labels_2):
        """ Transfers style of style images to content images. Assumes input
            is a Paddle tensor with a batch dimension."""
        B, sC, sH, sW = images_1.shape
        B, tC, tH, tW = images_2.shape
        if (sH != tH) or (sW != tW):
            images_1 = F.interpolate(images_1, size=(tH, tW), mode='bicubic')
            labels_1 = F.interpolate(
                labels_1, size=(tH, tW), mode='nearest')
        mixed_images, mixed_labels = self.cutmix(
            images_1, images_2, labels_1, labels_2, beta=self.beta)
        return mixed_images, mixed_labels


class MaskGenerator(object):
    """
    Mask Generator
    """

    def generate_params(self, n_masks, mask_shape, rng=None):
        raise NotImplementedError('Abstract')

    def append_to_batch(self, *batch):
        x,y = batch[0]
        params = self.generate_params(len(x), x.shape[2:4])
        return batch + (params,)

    def paddle_masks_from_params(self, t_params):
        raise NotImplementedError('Abstract')


def gaussian_kernels(sigma, max_sigma=None, truncate=4.0):
    """
    Generate multiple 1D gaussian convolution kernels
    :param sigma: values for sigma as a `(N,)` array
    :param max_sigma: maximum possible value for sigma or None to compute it; used to compute kernel size
    :param truncate: kernel size truncation factor
    :return: kernels as a `(N, kernel_size)` array
    """
    if max_sigma is None:
        max_sigma = sigma.max()
    sigma = sigma[:, None]
    radius = int(truncate * max_sigma + 0.5)
    sigma2 = sigma * sigma
    x = np.arange(-radius, radius + 1)[None, :]
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum(axis=1, keepdims=True)
    return phi_x


class BoxMaskGenerator(MaskGenerator):
    def __init__(self, prop_range, n_boxes=1, random_aspect_ratio=True, prop_by_area=True, within_bounds=True, invert=False):
        if isinstance(prop_range, float):
            prop_range = (prop_range, prop_range)
        self.prop_range = prop_range
        self.n_boxes = n_boxes
        self.random_aspect_ratio = random_aspect_ratio
        self.prop_by_area = prop_by_area
        self.within_bounds = within_bounds
        self.invert = invert

    def generate_params(self, n_masks, mask_shape, rng=None):
        """
        Box masks can be generated quickly on the CPU so do it there.
        >>> boxmix_gen = BoxMaskGenerator((0.25, 0.25))
        >>> params = boxmix_gen.generate_params(256, (32, 32))
        >>> t_masks = boxmix_gen.paddle_masks_from_params(params, (32, 32), 'cuda:0')
        :param n_masks: number of masks to generate (batch size)
        :param mask_shape: Mask shape as a `(height, width)` tuple
        :param rng: [optional] np.random.RandomState instance
        :return: masks: masks as a `(N, 1, H, W)` array
        """
        if rng is None:
            rng = np.random

        if self.prop_by_area:
            # Choose the proportion of each mask that should be above the threshold
            mask_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))

            # Zeros will cause NaNs, so detect and suppres them
            zero_mask = mask_props == 0.0

            if self.random_aspect_ratio:
                y_props = np.exp(rng.uniform(low=0.0, high=1.0, size=(n_masks, self.n_boxes)) * np.log(mask_props))
                x_props = mask_props / y_props
            else:
                y_props = x_props = np.sqrt(mask_props)
            fac = np.sqrt(1.0 / self.n_boxes)
            y_props *= fac
            x_props *= fac

            y_props[zero_mask] = 0
            x_props[zero_mask] = 0
        else:
            if self.random_aspect_ratio:
                y_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))
                x_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))
            else:
                x_props = y_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))
            fac = np.sqrt(1.0 / self.n_boxes)
            y_props *= fac
            x_props *= fac

        sizes = np.round(np.stack([y_props, x_props], axis=2) * np.array(mask_shape)[None, None, :])

        if self.within_bounds:
            positions = np.round((np.array(mask_shape) - sizes) * rng.uniform(low=0.0, high=1.0, size=sizes.shape))
            rectangles = np.append(positions, positions + sizes, axis=2)
        else:
            centres = np.round(np.array(mask_shape) * rng.uniform(low=0.0, high=1.0, size=sizes.shape))
            rectangles = np.append(centres - sizes * 0.5, centres + sizes * 0.5, axis=2)

        if self.invert:
            masks = np.zeros((n_masks, 1) + mask_shape)
        else:
            masks = np.ones((n_masks, 1) + mask_shape)
        for i, sample_rectangles in enumerate(rectangles):
            for y0, x0, y1, x1 in sample_rectangles:
                masks[i, 0, int(y0):int(y1), int(x0):int(x1)] = 1 - masks[i, 0, int(y0):int(y1), int(x0):int(x1)]
        return masks


class GaussianBlurLayer(nn.Layer):
    """ Add Gaussian Blur to a 4D tensor
    This layer takes a 4D tensor of {N, C, H, W} as input.
    The Gaussian blur will be performed in given channel number (C) splitly.
    """

    def __init__(self, channels, kernel_size):
        """
        Arguments:
            channels (int): Channel for input tensor
            kernel_size (int): Size of the kernel used in blurring
        """

        super(GaussianBlurLayer, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        assert self.kernel_size % 2 != 0

        self.op = nn.Sequential(
            nn.Pad2D(math.floor(self.kernel_size / 2)),
            nn.Conv2D(channels, channels, self.kernel_size,
                      stride=1, padding=0, bias_attr=False, groups=channels)
        )

        self._init_kernel()

    def forward(self, x):
        """
        Arguments:
            x (paddle.Tensor): input 4D tensor
        Returns:
            paddle.Tensor: Blurred version of the input
        """

        if not len(list(x.shape)) == 4:
            print('\'GaussianBlurLayer\' requires a 4D tensor as input\n')
        elif not x.shape[1] == self.channels:
            print('In \'GaussianBlurLayer\', the required channel ({0}) is'
                           'not the same as input ({1})\n'.format(self.channels, x.shape[1]))

        return self.op(x)

    def _init_kernel(self):
        sigma = 0.3 * ((self.kernel_size - 1) * 0.5 - 1) + 0.8
        n = np.zeros((self.kernel_size, self.kernel_size))
        i = math.floor(self.kernel_size / 2)
        n[i, i] = 1
        kernel = gaussian_filter(n, sigma)
        kernel = paddle.to_tensor(kernel)
        kernel = kernel.unsqueeze(0)
        kernel = kernel.unsqueeze(0)
        for name, param in self.named_parameters():
            param.set_value(kernel.astype(paddle.float32))


class ClassMaskGenerator(object):

    def generate_class_mask(self, pred, classes):
        if classes is not None:
            pred, classes = paddle.broadcast_tensors([pred.unsqueeze(0), classes.unsqueeze(1).unsqueeze(2)])
            N = pred.equal(classes).sum(0)
        else:
            N = paddle.zeros(shape=pred.shape[-2:]).astype(paddle.int64)
        return N

    def mask(self, batch_size, argmax_u_w, ignore_label):
        for image_i in range(batch_size):
            classes = paddle.unique(argmax_u_w[image_i])
            noigmask = classes != ignore_label
            if noigmask.shape[0] != 1:
                classes = classes[classes != ignore_label]
            nclasses = classes.shape[0]
            if nclasses > 1:
                classes = (classes[
                    paddle.to_tensor(
                        np.random.choice(nclasses, int((nclasses - nclasses % 2) / 2), replace=False)).astype(
                        paddle.int64)])
            else:
                classes = None
            if image_i == 0:
                MixMask = self.generate_class_mask(argmax_u_w[image_i], classes).unsqueeze(0)
            else:
                MixMask = paddle.concat(
                    (MixMask, self.generate_class_mask(argmax_u_w[image_i], classes).unsqueeze(0)))
        return MixMask

    def mix(self, mask, data=None):
        # Mix
        if not (data is None):
            if mask.shape[0] == data.shape[0]:
                data = paddle.concat(
                    [(mask[i] * data[i] + (1 - mask[i]) * data[(i + 1) % data.shape[0]]).unsqueeze(0) for i in
                     range(data.shape[0])])
            elif mask.shape[0] == data.shape[0] / 2:
                data = paddle.concat((paddle.concat(
                    [(mask[i] * data[2 * i] + (1 - mask[i]) * data[2 * i + 1]).unsqueeze(0) for i in
                     range(int(data.shape[0] / 2))]),
                                      paddle.concat(
                                          [((1 - mask[i]) * data[2 * i] + mask[i] * data[2 * i + 1]).unsqueeze(0) for i
                                           in
                                           range(int(data.shape[0] / 2))])))

        return data.astype(paddle.float32)