import cv2
import numpy as np
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
import datetime
import time



def EOG(img1,img2):  #输入退化图像和校正图像计算EOG提升比例
    I_1 = np.double(img1)
    M, N = I_1.shape
    FI_1 = 0
    for x in range(M-1):
        for y in range(N-1):
            #x方向和y方向的相邻像素灰度值只差的的平方和作为清晰度值
            FI_1 = FI_1 + (I_1[x + 1, y] - I_1[x, y]) * (I_1[x + 1, y] - I_1[x, y]) + (I_1[x, y + 1] - I_1[x, y]) * (
                 I_1[x, y + 1] - I_1[x, y])
    I_2 = np.double(img2)
    M, N = I_2.shape
    FI_2 = 0
    for x in range(M-1):
        for y in range(N-1):
            # x方向和y方向的相邻像素灰度值只差的的平方和作为清晰度值
            FI_2 = FI_2 + (I_2[x + 1, y] - I_2[x, y]) * (I_2[x + 1, y] - I_2[x, y]) + (I_2[x, y + 1]- I_2[x, y]) * (
                I_2[x, y + 1] - I_2[x, y])
    P = (FI_2 - FI_1) / FI_1
    return P
def evlauate_metrics(img1,img2,img3,t,name): #输入退化图像，校正图像以及原图，返回它们的评价指标
    #原图和退化图像的psnr1
    psnr1 = compare_psnr(img3, img1, data_range=255)
    #原图和校正图像的psnr2
    psnr2 = compare_psnr(img3, img2, data_range=255)

    #原图和退化图像的ssim
    ssim1 = compare_ssim(img3, img1, data_range=255)
    #原图和校正图像的ssim
    ssim2 = compare_ssim(img3, img2, data_range=255)

    #退化图像和校正图像的EOG差值，EOG-单图评价指标
    eog = EOG(img1, img2)

    #psnr,ssim,EOG差值
    diff_psnr = psnr2 - psnr1
    diff_ssim = (ssim2 - ssim1)/ssim1
    #输出参数至txt文本上，分别是PSNR（退化、校正），ssmi（退化、校正）
    #                        EOG提升比例、时间（s）、PSNR提升，SSMI提升比例
    f = open('metrics.txt','a+')
    f.write(name+' '+str(psnr1)+' '+str(psnr2)+' '+str(ssim1)+' '+str(ssim2)
            +' '+str(eog)+' '+str(t)+' '+str(diff_psnr)+' '+str(diff_ssim)+'\n')
    f.closed

