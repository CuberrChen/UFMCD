#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/12/27 下午8:43
# @Author  : chenxb
# @FileName: csv_reader.py
# @Software: PyCharm
# @mail ：joyful_chen@163.com


import csv
import numpy as np
import matplotlib.pyplot as plt



def moving_average(interval, windowsize):
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'same')
    return re

x = []
y = []
with open('visualdl-scalar-SemiSeg-Paddle1212_advSemiSeg_deeplabv2_res101_vocaug_0.5_20k-Train_loss_seg_value.csv','r') as f:
    reader = csv.reader(f)
    for i in reader:
        print(i)
        if i[0]!='id':
            x.append(int(i[0]))
            y.append(float(i[3]))

## 绘制多条曲线在同一个fig
step = np.array(x)
loss = np.array(y)

plt.plot(x, y, 'k')  # plot(横坐标，纵坐标， 颜色)
plt.plot(step,loss)
y_av = moving_average(y, 6)
plt.plot(x, y_av, 'b')
# plt.grid()网格线设置
# plt.grid(True)
plt.title("AdvSemiSeg Training on Pascal2012Aug")
plt.xlabel("Step")
plt.ylabel("Loss")
# plt.savefig('foo.eps', format='eps', dpi=1000) # 矢量图
plt.show()
