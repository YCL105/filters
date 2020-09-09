import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os

#中间过程可视化
def visualization(tensor,epoch,name):
    if not os.path.isdir('visualization'):
        os.mkdir('visualization')
    a,b,m,n = tensor.shape
    for d in range(a):
        x1 = tensor[d,:,:,:]
        x1 = x1.reshape(b,m,n)
        x1 = x1.cpu().detach().numpy()
        xx = np.transpose(x1,(1,2,0))

        cv2.imwrite('./visualization/'+name+str(epoch)+'_'+str(d)+'.jpg',xx*255.)




#设计模糊核拼接
def change_blur(tensor):
    flag = 1  #防止模糊核全部输出0，有存在过，望以后可以解决
    a,b,m,n = tensor.shape #取出tensor的shape
    #tensor = tensor.reshape(m,n)
    blur = (torch.zeros([64,64])).cuda()
    criter = (torch.zeros([64,64])).cuda()
    pre_blur = torch.zeros([64,64])
    pre_blur = pre_blur.cuda()

    for d in range(a):  #某个批次
        tt = tensor[d,:,:,:] #批次，通道，高，宽
        tt = tt.reshape(m,n)

        for i in range(m):  #有多少的通道
            for j in range(n):

                for x in range(64): #扩充数据32*32
                    for y in range(64):
                        pre_blur[x,y] = tt[i,j]
                if torch.equal(blur,criter) & flag:
                    blur = pre_blur
                    flag = 0
                else:
                    blur = torch.cat((blur,pre_blur))
    blur = blur.reshape([a,m*n,64,64])
    return blur

#基础结构
class Net(nn.Module):
    def __init__(self,downsample=None):
        super(Net, self).__init__()
        #卷积层提取模糊核信息
        self.conv1 = nn.Conv2d(1,16,(3,3),stride=2,padding=1)#128
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, (3, 3), stride=1, padding=1)  # 64
        self.conv2_1 = nn.Conv2d(32, 64,(5,5),stride=2,padding=2) #64
        self.conv3 = nn.Conv2d(64,4,(3,3),stride=2,padding=1) #32
        self.conv4 = nn.Conv2d(4, 30, (3, 3), stride=2, padding=1)  # 16
        self.conv5 = nn.Conv2d(30, 78, (5, 5), stride=2, padding=2)  # 8
        self.conv6 = nn.Conv2d(78, 8, (3, 3), stride=1, padding=1)  # 8
        self.conv7 = nn.Conv2d(8,1,(1,1),1)
        #resnet层恢复图像
        self.res1_conv1 = nn.Conv2d(128, 32, (3, 3), 1,padding=1)#8*8+Conv2
        self.res1_bn1 = nn.BatchNorm2d(32)
        self.res1_conv2 = nn.Conv2d(32, 64, (3, 3), 1,padding=1)
        self.res1_bn2 = nn.BatchNorm2d(64)
        self.conv1x1_1 = nn.Conv2d(128, 64, (1, 1), 1)

        #resnet层恢复图像
        self.res2_conv1 = nn.Conv2d(128,50, (3, 3), 1,padding=1)
        self.res2_bn1 = nn.BatchNorm2d(50)
        self.res2_conv2 = nn.Conv2d(50, 27, (3, 3), 1,padding=1)
        self.res2_bn2 = nn.BatchNorm2d(27)
        self.res2_conv3 = nn.Conv2d(27, 64, (3, 3), 1, padding=1)
        self.res2_bn3 = nn.BatchNorm2d(64)
        self.conv1x1_2 = nn.Conv2d(128, 64, (1, 1), 1)

        # resnet层恢复图像
        self.res3_conv1 = nn.Conv2d(128, 40, (3, 3), 1, padding=1)
        self.res3_bn1 = nn.BatchNorm2d(40)
        self.res3_conv2 = nn.Conv2d(40, 64, (3, 3), 1, padding=1)
        self.res3_bn2 = nn.BatchNorm2d(64)
        self.conv1x1_3 = nn.Conv2d(128, 64, (1, 1), 1)

        #第一次上采样后卷积
        self.up_conv1 = nn.Conv2d(128,30,(3,3),1,padding=1)
        self.up_conv2 = nn.Conv2d(30,16,(1,1),1)
        self.upsample_1 = nn.PixelShuffle(4)
        self.up_conv3 = nn.Conv2d(1,8,(3,3),1,padding=1)
        self.up_conv4 = nn.Conv2d(8,1,(1,1),1)



    # 激活函数既可以使用nn，又可以调用nn.functional
    def forward(self, x_1,epoch,name):

        #模糊图片输入，提取模糊核信息
        out = self.conv1(x_1)#128

        out = F.relu(out)  # # 激活函数，直接调用torch.nn.functional中集成好的Relu
        out = self.conv2(out) #64
        out = self.conv2_1(out)
        out = F.relu(out)
        downsample_x1 = out  # 64
        out = self.conv3(out)
        out = F.relu(out)
        out = self.conv4(out)
        out = F.relu(out)
        out = self.conv5(out)
        out = F.relu(out)
        out = self.conv6(out)
        out = F.relu(out)
        out = self.conv7(out)
        out = F.relu(out)#8*8

        #模糊核信息和初步提取的模糊图片信息拼接
        blur = change_blur(out)
        out = torch.cat((blur,downsample_x1),dim=1) #模糊核一维扩展和原图卷积拼接
        #恢复图像resnetblock块
        #模糊核拼接，卷积，原图相加
        residual_1 = out

        out = self.res1_conv1(out)
        out = self.res1_bn1(out)
        out = F.relu(out)

        out = self.res1_conv2(out)
        out = self.res1_bn2(out)
        residual_1 = self.conv1x1_1(residual_1)

        out = out + residual_1 + downsample_x1

        # 第二个resnet块
        # 模糊核拼接，卷积，原图相加
        out = torch.cat((blur, out), dim=1)
        residual_2 = out
        out = self.res2_conv1(out)
        out = self.res2_bn1(out)
        out = F.relu(out)
        out = self.res2_conv2(out)
        out = self.res2_bn2(out)
        out = F.relu(out)

        out = self.res2_conv3(out)
        out = self.res2_bn3(out)
		
        residual_2 = self.conv1x1_2(residual_2)

        out = out + residual_2 + downsample_x1


        # 第三个resnet块
        # 模糊核拼接，卷积，原图相加
        out = torch.cat((blur, out), dim=1)
        residual_3 = out
        out = self.res3_conv1(out)
        out = self.res3_bn1(out)
        out = F.relu(out)

        out = self.res3_conv2(out)
        out = self.res3_bn2(out)
		
        residual_3 = self.conv1x1_3(residual_3)


        out = out + residual_3 + downsample_x1

        #上采样

        out = torch.cat((blur, out), dim=1) #拼接

        out = self.up_conv1(out)
        out = F.relu(out)
        out = self.up_conv2(out)
        out = F.relu(out)
        out = self.upsample_1(out)
        out = torch.sigmoid(out)
        #visualization(out, epoch, name)





        return out
