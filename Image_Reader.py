import torch
import os
import numpy as np
import cv2

#import matplotlib as plt
#from test import *


# 读取图片的函数，接收六个参数
# 输入参数分别是训练集或测试集路径，图片格式，标签格式，需要调整的尺寸大小
def ImageReader(file_name, size=256):
    x_data = []
    y_data = []

    data_file = file_name + '/data/'  # 得到图片名称和路径
    label_file = file_name + '/label/'  # 得到标签名称和路径
    list1 = ['aaa' for _ in range(len(os.listdir(data_file)))] #创建一个字符串列表

    a = 0
    for picture_name in os.listdir(data_file):
         #保存图片名称，后续可视化会用该名称命名
        list1[a] = picture_name
        a += 1

        picture = cv2.imread(data_file+picture_name, 0)  # 读取图片
        #height = picture.shape[0]  # 得到图片的高
        #width = picture.shape[1]  # 得到图片的宽
        picture_resize_t = cv2.resize(picture, (size, size))  # 调整图片的尺寸，改变成网络输入的大小
        picture_resize = picture_resize_t / 255.  # 归一化图片
        if not x_data == []:
            picture_resize = picture_resize.reshape(1,1,size,size)
            x_data = np.concatenate((x_data,picture_resize),axis=0)
        else:
            x_data = picture_resize.reshape(1,1,size,size)
    for label_name in os.listdir(label_file):
        label = cv2.imread(label_file+label_name, 0)  # 读取标签
       # height = label.shape[0]  # 得到图片的高
       # width = label.shape[1]  # 得到图片的宽
        label_resize_t = cv2.resize(label, (size, size))  # 调整标签的尺寸，改变成网络输入的大小
        label_resize = label_resize_t / 255.  # 归一化标签

        if not y_data == []:
            label_resize = label_resize.reshape(1,1,size,size)
            y_data = np.concatenate((y_data, label_resize), axis=0)
        else:
            y_data = label_resize.reshape(1,1,size,size)

    xx = torch.from_numpy(x_data) #B,C,H,W
    yy = torch.from_numpy(y_data)

    xx = xx.type(torch.FloatTensor)
    yy = yy.type(torch.FloatTensor)
    return xx,yy ,list1 # 返回网络输入的图片，标签，还有原图片和标签的长宽
#x,y,l = ImageReader('./test/')
#print(l,l[0],l[1],l[2])
#x1 = x[1,:,:,:]
#print(type(x1))
#x1 = torch.from_numpy(x1)
#print(x1.shape)
#xx = np.transpose(x1.numpy(),(1,2,0))
#print(x.shape)
#xx = x1.numpy()
#xx = xx.reshape(256,256,1)
#print(type(xx))
#cv2.imwrite('./x01.jpg',xx*255.)
#print(xx*255)
#y1 = y[1,:,:,:]
#!y1 = y1.reshape(256,256,1)
#cv2.imwrite('./y1.jpg',y1*255.)
#os.system('pause')
