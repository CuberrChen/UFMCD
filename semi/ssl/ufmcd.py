#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/1/13 下午6:20
# @Author  : chenxb
# @FileName: cps.py
# @Software: PyCharm
# @mail ：joyful_chen@163.com

import os
import time
import copy
import shutil
import numpy as np
from collections import deque

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.utils import (TimeAverager, calculate_eta, resume, logger,
                             worker_init_fn, op_flops_funs)
from paddleseg.core.val import evaluate

from semi.utils.dataloader import getDataLoader
from semi.configs.config import config
from semi.models import WrappedModel
from semi.utils.ramps import sigmoid_rampup
from semi.utils.augmentation import BoxMaskGenerator

NUM_CLASSES = config.NUM_CLASSES  # 21 for PASCAL-VOC / 60 for PASCAL-Context / 19 Cityscapes
MODEL = config.MODEL_NAME
SUP_WEIGHT = config.UFMCD.SUP_WEIGHT
CPS_WEIGHT = config.UFMCD.CPS_WEIGHT
RAMPUP_LENGTH = config.UFMCD.RAMPUP_LENGTH
USE_CUTMIX = config.UFMCD.USE_CUTMIX
MASK_PROP_RANGE = config.UFMCD.MASK_PROP_RANGE
BOXMASK_N_BOXES = config.UFMCD.BOXMASK_N_BOXES
BOXMASK_FIXED_ASPECT_RATIO = config.UFMCD.BOXMASK_FIXED_ASPECT_RATIO
BOXMASK_BY_SIZE = config.UFMCD.BOXMASK_BY_SIZE
BOXMASK_OUTSIDE_BOUNDS = config.UFMCD.BOXMASK_OUTSIDE_BOUNDS
BOXMASK_NO_INVERT = config.UFMCD.BOXMASK_NO_INVERT
ADEMT_WEIGHT = 1.0
CROSS_WEIGHT = 1.0
threhold = 0.80
CPS_WEIGHT_0 = 1
alpha = 0.5

def check_logits_losses(logits_list, losses):
    len_logits = len(logits_list)
    len_losses = len(losses['types'])
    if len_logits != len_losses:
        raise RuntimeError(
            'The length of logits_list should equal to the types of loss config: {} != {}.'
                .format(len_logits, len_losses))


def loss_computation(logits_list, labels, losses, edges=None):
    check_logits_losses(logits_list, losses)
    loss_list = []
    for i in range(len(logits_list)):
        logits = logits_list[i]
        loss_i = losses['types'][i]
        # Whether to use edges as labels According to loss type.
        if loss_i.__class__.__name__ in ('BCELoss',
                                         'FocalLoss') and loss_i.edge_label:
            loss_list.append(losses['coef'][i] * loss_i(logits, edges))
        elif loss_i.__class__.__name__ in ("KLLoss",):
            loss_list.append(losses['coef'][i] * loss_i(
                logits_list[0], logits_list[1].detach()))
        else:
            loss_list.append(losses['coef'][i] * loss_i(logits, labels))
    return loss_list


def train_UFMCD(
        cfg,
        model,
        train_dataset,
        label_ratio,
        split_id,
        semi_start_iter,
        val_dataset,
        optimizer,
        save_dir,
        iters,
        batch_size,
        resume_model,
        save_interval,
        log_iters,
        num_workers,
        use_vdl,
        losses,
        keep_checkpoint_max,
        test_config,
        fp16,
        profiler_options,
        to_static_training,
        case
):
    # 环境构建
    nranks = paddle.distributed.ParallelEnv().nranks
    local_rank = paddle.distributed.ParallelEnv().local_rank

    if not os.path.isdir(save_dir):
        if os.path.exists(save_dir):
            os.remove(save_dir)
        os.makedirs(save_dir)

    # copy train_ssl.py file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(os.path.join(this_dir, os.path.basename(__file__)), save_dir)

    # 模型构建
    start_iter = 0
    TrainModel = WrappedModel(model)
    TrainModel.train()
    # add aux optimizer
    optimizer_r = paddle.optimizer.Momentum(parameters=TrainModel.model_r.parameters(),
                                            learning_rate=copy.deepcopy(optimizer._learning_rate),
                                            momentum=copy.deepcopy(optimizer._momentum),
                                            weight_decay=copy.deepcopy(optimizer._regularization_coeff))
    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
    mse_criterion = nn.MSELoss()
    if resume_model is not None:
        start_iter = TrainModel.resume(optimizer, optimizer_r, resume_model)

    if nranks > 1:
        paddle.distributed.fleet.init(is_collective=True)
        optimizer = paddle.distributed.fleet.distributed_optimizer(
            optimizer)  # The return is Fleet object
        optimizer_r = paddle.distributed.fleet.distributed_optimizer(
            optimizer_r)  # The return is Fleet object
        ddp_model = paddle.distributed.fleet.distributed_model(TrainModel)

    # 数据准备
    trainloader, trainloader_remain, trainloader_gt, iters_per_epoch = getDataLoader(dataset=train_dataset,
                                                                                     batch_size=batch_size,
                                                                                     num_workers=num_workers,
                                                                                     label_ratio=label_ratio,
                                                                                     split_id=split_id,
                                                                                     save_dir=save_dir,
                                                                                     worker_init_fn=worker_init_fn)
    if USE_CUTMIX:
        trainloader_plus, trainloader_remain_plus, _, _ = getDataLoader(dataset=train_dataset,
                                                                        batch_size=batch_size,
                                                                        num_workers=num_workers,
                                                                        label_ratio=label_ratio,
                                                                        split_id=split_id,
                                                                        save_dir=save_dir,
                                                                        worker_init_fn=worker_init_fn)
    trainloader_iter = iter(trainloader)
    trainloader_remain_iter = iter(trainloader_remain)
    if USE_CUTMIX:
        trainloader_remain_plus_iter = iter(trainloader_remain_plus)

        mask_generator = BoxMaskGenerator(prop_range=MASK_PROP_RANGE, n_boxes=BOXMASK_N_BOXES,
                                          random_aspect_ratio=not BOXMASK_FIXED_ASPECT_RATIO,
                                          prop_by_area=not BOXMASK_BY_SIZE, within_bounds=not BOXMASK_OUTSIDE_BOUNDS,
                                          invert=not BOXMASK_NO_INVERT)

    if use_vdl:
        from visualdl import LogWriter
        log_writer = LogWriter(save_dir)

    # 训练
    best_mean_iou = -1.0
    best_model_iter = -1
    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    save_models = deque()
    batch_start = time.time()
    i_iter = start_iter
    while i_iter < iters:
        i_iter += 1
        rampval = sigmoid_rampup(i_iter, rampup_length=RAMPUP_LENGTH)
        loss_l_value = 0
        loss_r_value = 0
        loss_unsup_value = 0

        optimizer.clear_grad()
        optimizer_r.clear_grad()
        # training loss for labeled data only
        try:
            batch = next(trainloader_iter)
        except:
            trainloader_iter = iter(trainloader)
            batch = next(trainloader_iter)

        images, labels = batch[0], batch[1].astype('int64')
        if hasattr(model, 'data_format') and model.data_format == 'NHWC':
            images = images.transpose((0, 2, 3, 1))

        # traing on unlabeled data
        try:
            batch_remain = next(trainloader_remain_iter)
            if USE_CUTMIX:
                batch_remain_plus = next(trainloader_remain_plus_iter)
        except:
            trainloader_remain_iter = iter(trainloader_remain)
            batch_remain = next(trainloader_remain_iter)
            if USE_CUTMIX:
                trainloader_remain_plus_iter = iter(trainloader_remain_plus)
                batch_remain_plus = next(trainloader_remain_plus_iter)

        images_unlabel = batch_remain[0]
        if USE_CUTMIX:
            images_unlabel_plus = batch_remain_plus[0]
        if hasattr(model, 'data_format') and model.data_format == 'NHWC':
            images_unlabel = images_unlabel.transpose((0, 2, 3, 1))
            if USE_CUTMIX:
                images_unlabel_plus = images_unlabel_plus.transpose((0, 2, 3, 1))

        reader_cost_averager.record(time.time() - batch_start)

        ## get cutmix mask
        batch_mix_masks = mask_generator.generate_params(n_masks=images_unlabel.shape[0],
                                                         mask_shape=tuple(images_unlabel.shape[2:4]))
        ## Mix images with masks
        unsup_imgs_mixed = images_unlabel * paddle.to_tensor((1 - batch_mix_masks)) + \
                           images_unlabel_plus * paddle.to_tensor(batch_mix_masks)

        ###############################CPNSP##############################################
        with paddle.no_grad():
            logits_u0_tea_1 = TrainModel(images_unlabel, step=1)[0]
            logits_u1_tea_1 = TrainModel(images_unlabel_plus, step=1)[0]
            # Estimate the pseudo-label with branch#2 & supervise branch#1
            logits_u0_tea_2 = TrainModel(images_unlabel, step=2)[0]
            logits_u1_tea_2 = TrainModel(images_unlabel_plus, step=2)[0]
            logits_u0_tea_1 = logits_u0_tea_1.detach()
            logits_u1_tea_1 = logits_u1_tea_1.detach()
            logits_u0_tea_2 = logits_u0_tea_2.detach()
            logits_u1_tea_2 = logits_u1_tea_2.detach()

        # construct mix logit and postive learning and negative learning
        logits_cons_tea_1 = logits_u0_tea_1 * paddle.to_tensor((1 - batch_mix_masks)) + \
                            logits_u1_tea_1 * paddle.to_tensor(batch_mix_masks)
        ps_label_1 = paddle.argmax(logits_cons_tea_1, axis=1).astype(paddle.int64)
        ns_label_1 = paddle.argmin(logits_cons_tea_1, axis=1).astype(paddle.int64) # negative learning
        with paddle.no_grad():
            prob_cons_tea_1 = F.softmax(logits_cons_tea_1, axis=1)
            entropy = -paddle.sum(prob_cons_tea_1 * paddle.log(prob_cons_tea_1 + 1e-10), axis=1)
            w_1 = 1 - entropy / paddle.log(paddle.to_tensor(prob_cons_tea_1.shape[1],dtype=paddle.float32)) # refer to cls
            threhold_1 = threhold
            mask_1 = w_1 > threhold_1
            mask_1.stop_gradient = True

        logits_cons_tea_2 = logits_u0_tea_2 * paddle.to_tensor((1 - batch_mix_masks)) + \
                            logits_u1_tea_2 * paddle.to_tensor(batch_mix_masks)
        ps_label_2 = paddle.argmax(logits_cons_tea_2, axis=1).astype(paddle.int64)
        ns_label_2 = paddle.argmin(logits_cons_tea_2, axis=1).astype(paddle.int64) # negative learning
        with paddle.no_grad():
            prob_cons_tea_2 = F.softmax(logits_cons_tea_2, axis=1)
            entropy = -paddle.sum(prob_cons_tea_2 * paddle.log(prob_cons_tea_2 + 1e-10), axis=1)
            w_2 = 1 - entropy / paddle.log(paddle.to_tensor(prob_cons_tea_2.shape[1],dtype=paddle.float32)) # refer to cls归一化， 并求反，值越大表示越稳定。
            threhold_2 = threhold
            mask_2 = w_2 > threhold_2
            mask_2.stop_gradient = True

        if nranks > 1:
            # supervised loss on both models
            pred_sup_l = ddp_model(images, step=1)
            pred_sup_r = ddp_model(images, step=2)
        else:
            # supervised loss on both models
            pred_sup_l, feature_l = TrainModel(images, step=3)
            pred_sup_r, feature_r = TrainModel(images, step=4)

        lam = np.random.beta(alpha,alpha)
        lam = min(lam, 1 - lam)
        logits_cons_stu_1 = TrainModel(unsup_imgs_mixed, step=5, feature=feature_l, lam=lam)
        # Get student#2 prediction for mixed image
        logits_cons_stu_2 = TrainModel(unsup_imgs_mixed, step=6, feature=feature_r, lam=lam)
        # soft cross-decouple
        logits_cons_stu_1_decouple = logits_cons_stu_1[0] - lam * pred_sup_l[0]
        logits_cons_stu_1_decouple_ = logits_cons_stu_1[0] - lam * pred_sup_r[0]
        logits_cons_stu_2_decouple = logits_cons_stu_2[0] - lam * pred_sup_r[0]
        logits_cons_stu_2_decouple_ = logits_cons_stu_2[0] - lam * pred_sup_l[0]

        decouple_consist_loss = mse_criterion(logits_cons_stu_1_decouple, logits_cons_stu_1_decouple_) + \
                                mse_criterion(logits_cons_stu_2_decouple, logits_cons_stu_2_decouple_)
        logits_cons_stu_1 = (logits_cons_stu_1_decouple + logits_cons_stu_1_decouple_) / 2.
        logits_cons_stu_2 = (logits_cons_stu_2_decouple + logits_cons_stu_2_decouple_) / 2.
        logits_cons_stu_1 = paddle.transpose(logits_cons_stu_1, [0, 2, 3, 1])
        logits_cons_stu_2 = paddle.transpose(logits_cons_stu_2, [0, 2, 3, 1])

        loss_cross_l = paddle.mean(w_2 * criterion(logits_cons_stu_1, ps_label_2) * mask_2) / paddle.mean(mask_2.astype(paddle.float32) + 1e-8)
        loss_cross_r = paddle.mean(w_1 * criterion(logits_cons_stu_2, ps_label_1) * mask_1) / paddle.mean(mask_1.astype(paddle.float32) + 1e-8)
        cps_loss_list = CROSS_WEIGHT * (loss_cross_l + loss_cross_r) + ADEMT_WEIGHT * decouple_consist_loss
        cps_loss = cps_loss_list * CPS_WEIGHT

        loss_sup_list = loss_computation(pred_sup_l, labels, losses)
        loss_sup = sum(loss_sup_list)

        loss_sup_r_list = loss_computation(pred_sup_r, labels, losses)
        loss_sup_r = sum(loss_sup_r_list)

        loss = loss_sup + loss_sup_r + cps_loss
        loss.backward()

        loss_l_value = loss_sup.detach().cpu().numpy()[0]
        loss_r_value = loss_sup_r.detach().cpu().numpy()[0]
        loss_unsup_value = cps_loss.detach().cpu().numpy()[0]
        loss_decouple_consist_value = decouple_consist_loss.detach().cpu().numpy()[0]

        optimizer.step()
        optimizer_r.step()
        lr = optimizer.get_lr()
        lr_r = optimizer_r.get_lr()
        # update lr
        if isinstance(optimizer, paddle.distributed.fleet.Fleet):
            lr_sche = optimizer.user_defined_optimizer._learning_rate
            lr_r_sche = optimizer_r.user_defined_optimizer._learning_rate
        else:
            lr_sche = optimizer._learning_rate
            lr_r_sche = optimizer_r._learning_rate
        if isinstance(lr_sche, paddle.optimizer.lr.LRScheduler):
            lr_sche.step()
            lr_r_sche.step()

        batch_cost_averager.record(
            time.time() - batch_start, num_samples=batch_size)
        if (i_iter) % log_iters == 0 and local_rank == 0:
            remain_iters = iters - i_iter
            avg_train_batch_cost = batch_cost_averager.get_average()
            avg_train_reader_cost = reader_cost_averager.get_average()
            eta = calculate_eta(remain_iters, avg_train_batch_cost)
            logger.info(
                "[TRAIN] epoch: {}, iter: {}/{}, loss_l_value: {:.4f},"
                "loss_r_value: {:.4f}, loss_unsup_value: {:.4f},"
                " lr: {:.6f}, batch_cost: {:.4f}, reader_cost: {:.5f}, ips: {:.4f} samples/sec | ETA {}"
                    .format((i_iter - 1) // iters_per_epoch + 1, i_iter, iters,
                            loss_l_value, loss_r_value, loss_unsup_value,
                            lr, avg_train_batch_cost,
                            avg_train_reader_cost,
                            batch_cost_averager.get_ips_average(), eta))
            if use_vdl:
                log_writer.add_scalar('Train/loss_l', loss_l_value, i_iter)
                log_writer.add_scalar('Train/loss_r', loss_r_value, i_iter)
                log_writer.add_scalar('Train/loss_unsup_value', loss_unsup_value, i_iter)
                log_writer.add_scalar('Train/loss_decouple_consist_value', loss_decouple_consist_value, i_iter)
                log_writer.add_scalar('Train/lr', lr, i_iter)
                log_writer.add_scalar('Train/batch_cost',
                                      avg_train_batch_cost, i_iter)
                log_writer.add_scalar('Train/reader_cost',
                                      avg_train_reader_cost, i_iter)
        reader_cost_averager.reset()
        batch_cost_averager.reset()
        # 评估
        if (i_iter % save_interval == 0
            or i_iter == iters) and (val_dataset is not None):
            num_workers = 1 if num_workers > 0 else 0

            if test_config is None:
                test_config = {}

            mean_iou, acc, _, _, _ = evaluate(
                model, val_dataset, num_workers=num_workers, **test_config)
            model.train()

            print("Model R  Evaluate\n")
            mean_iou_r, acc_r, _, _, _ = evaluate(
                TrainModel.model_r, val_dataset, num_workers=num_workers, **test_config)
            TrainModel.model_r.train()

        if (i_iter % save_interval == 0 or i_iter == iters) and local_rank == 0:
            current_save_dir = os.path.join(save_dir,
                                            "iter_{}".format(i_iter))
            if not os.path.isdir(current_save_dir):
                os.makedirs(current_save_dir)
            paddle.save(TrainModel.state_dict(),
                        os.path.join(current_save_dir, 'model.pdparams'))
            optimizer_dict = {'optim_l': optimizer.state_dict(),
                              'optim_r': optimizer_r.state_dict()}
            paddle.save(optimizer_dict,
                        os.path.join(current_save_dir, 'model.pdopt'))
            save_models.append(current_save_dir)
            if len(save_models) > keep_checkpoint_max > 0:
                model_to_remove = save_models.popleft()
                shutil.rmtree(model_to_remove)

            if val_dataset is not None:
                if mean_iou > best_mean_iou and mean_iou > mean_iou_r:
                    best_mean_iou = mean_iou
                    best_model_iter = i_iter
                    best_model_dir = os.path.join(save_dir, "best_model")
                    paddle.save(
                        model.state_dict(),
                        os.path.join(best_model_dir, 'model.pdparams'))
                if mean_iou_r > best_mean_iou and mean_iou_r > mean_iou:
                    best_mean_iou = mean_iou_r
                    best_model_iter = i_iter
                    best_model_dir = os.path.join(save_dir, "best_model")
                    paddle.save(
                        TrainModel.model_r.state_dict(),
                        os.path.join(best_model_dir, 'model.pdparams'))

                logger.info(
                    '[EVAL] The model with the best validation mIoU ({:.4f}) was saved at iter {}.'
                        .format(best_mean_iou, best_model_iter))

                if use_vdl:
                    log_writer.add_scalar('Evaluate/mIoU_l', mean_iou, i_iter)
                    log_writer.add_scalar('Evaluate/Acc_l', acc, i_iter)
                    log_writer.add_scalar('Evaluate/mIoU_r', mean_iou_r, i_iter)
                    log_writer.add_scalar('Evaluate/Acc_r', acc_r, i_iter)
        batch_start = time.time()

    # Calculate flops.
    if local_rank == 0:
        _, c, h, w = images.shape
        _ = paddle.flops(
            model, [1, c, h, w],
            custom_ops={paddle.nn.SyncBatchNorm: op_flops_funs.count_syncbn})

    # Sleep for half a second to let dataloader release resources.
    time.sleep(0.5)
    if use_vdl:
        log_writer.close()
