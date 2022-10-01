#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/12/2 上午10:20
# @Author  : chenxb
# @FileName: dataloader.py
# @Software: PyCharm
# @mail ：joyful_chen@163.com
import os
import pickle
import numpy as np

import paddle
from paddleseg.transforms import RandomDistort

def getDataLoader(dataset,batch_size,num_workers,label_ratio,split_id,save_dir,worker_init_fn,remain_use_weak_aug=False,return_list=True,use_shared_memory=False):
    """
    split data. reeturn sub dataloder.
    Returns:

    """
    # 数据准备
    if label_ratio is None:
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        trainloader = paddle.io.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            return_list=return_list,
            use_shared_memory=use_shared_memory,
        )
        trainloader_gt = paddle.io.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            return_list=return_list,
            use_shared_memory=use_shared_memory,
        )
        trainloader_remain = paddle.io.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            return_list=return_list,
            use_shared_memory=use_shared_memory,
        )
        iters_per_epoch = len(batch_sampler)
    else:
        # sample partial data
        dataset_size = len(dataset)
        partial_size = int(label_ratio * dataset_size)

        if split_id is not None:
            train_ids = pickle.load(open(split_id, 'rb'))
            print('loading train ids from {}'.format(split_id))
        else:
            train_ids = np.arange(dataset_size)
            np.random.shuffle(train_ids)
        pickle.dump(train_ids, open(os.path.join(save_dir, 'train_id.pkl'), 'wb'))

        # split sub-dataset
        train_label_subset = paddle.io.Subset(dataset,indices=train_ids[:partial_size])
        train_remain_subset = paddle.io.Subset(dataset,indices=train_ids[partial_size:])
        train_gt_subset = paddle.io.Subset(dataset,indices=train_ids[:partial_size])

        if remain_use_weak_aug:
            weak_aug_trans = []
            for trans in train_remain_subset.dataset.transforms.transforms:
                if isinstance(trans, RandomDistort): # etal aug can remove by this
                    continue
                weak_aug_trans.append(trans)
            train_remain_subset.dataset.transforms.transforms = weak_aug_trans

        train_batch_sampler = paddle.io.DistributedBatchSampler(
            train_label_subset, batch_size=batch_size, shuffle=True, drop_last=True)
        train_remain_batch_sampler = paddle.io.DistributedBatchSampler(
            train_remain_subset, batch_size=batch_size, shuffle=True, drop_last=True)
        train_gt_batch_sampler = paddle.io.DistributedBatchSampler(
            train_gt_subset, batch_size=batch_size, shuffle=True, drop_last=True)

        # sub dataloader
        trainloader = paddle.io.DataLoader(train_label_subset,
                                           batch_sampler=train_batch_sampler,
                                           num_workers=num_workers,
                                           worker_init_fn=worker_init_fn,
                                           return_list=return_list,
                                           use_shared_memory=use_shared_memory)
        trainloader_remain = paddle.io.DataLoader(train_remain_subset,
                                                  batch_sampler=train_remain_batch_sampler,
                                                  num_workers=num_workers,
                                                  worker_init_fn=worker_init_fn,
                                                  return_list=return_list,
                                                  use_shared_memory=use_shared_memory)
        trainloader_gt = paddle.io.DataLoader(train_gt_subset,
                                              batch_sampler=train_gt_batch_sampler,
                                              num_workers=num_workers,
                                              worker_init_fn=worker_init_fn,
                                              return_list=return_list,
                                              use_shared_memory=use_shared_memory)
        iters_per_epoch = len(train_batch_sampler)

    return trainloader, trainloader_remain, trainloader_gt, iters_per_epoch
