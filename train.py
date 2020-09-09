import torch
import torch.nn as nn
import os
from torch.autograd import  Variable
from torch.utils.data import DataLoader,TensorDataset


from new_net import *
from Image_Reader import *
from test  import *

#超参数设置
batch_size = 5
epochs = 201
learning_rate = 0.001
use_cuda = torch.cuda.is_available()

#读取数据 dataloader类读取
train_path = './train/'
test_path = './test/'
train_data,train_label,_ = ImageReader(train_path)

train_dataset = TensorDataset(train_data,train_label)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

test_data,test_label,_ = ImageReader(test_path)

test_dataset = TensorDataset(test_data,test_label)
test_loader = DataLoader(dataset=test_dataset,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=0)






#基础结构
# class Net(nn.Module):
#     def __init__(self,downsample=None):
#         super(Net, self).__init__()
#         #卷积层提取模糊核信息
#         self.conv1 = nn.Conv2d(1, 16,(3,3),1,padding=1)
#         self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.conv2 = nn.Conv2d(16, 100,(5,5),1)
#         self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
#         self.conv3 = nn.Conv2d(100, 32, (3, 3), 1,padding=1)
#         self.pool3 = nn.MaxPool2d(kernel_size=2,stride=2,padding=1)
#         self.conv4 = nn.Conv2d(32,9,(3,3),1,padding=1)
#         self.conv5 = nn.Conv2d(9,1,(1,1),1)
#         #resnet层恢复图像
#         self.res1_conv1 = nn.Conv2d(82, 256, (3, 3), 1,padding=1)
#         self.res1_bn1 = nn.BatchNorm2d(256)
#         self.res1_conv2 = nn.Conv2d(256, 32, (3, 3), 1,padding=1)
#         self.res1_bn2 = nn.BatchNorm2d(32)
#         self.conv1x1_1 = nn.Conv2d(82,32,(1,1),1)
#         #resnet层恢复图像
#         self.res2_conv1 = nn.Conv2d(32,128, (3, 3), 1,padding=1)
#         self.res2_bn1 = nn.BatchNorm2d(128)
#         self.res2_conv2 = nn.Conv2d(128, 64, (3, 3), 1,padding=1)
#         self.res2_bn2 = nn.BatchNorm2d(64)
#         self.conv1x1_2 = nn.Conv2d(32,64, (1, 1),1)
#         #第一次上采样后卷积
#         self.upsample_1 = nn.PixelShuffle(4)
#         self.up_conv1 = nn.Conv2d(4,16,(3,3),1,padding=1)
#         self.up_conv2 = nn.Conv2d(16,64,(1,1),1)
#
#         #第二次上采样后卷积
#         self.upsample_2 = nn.PixelShuffle(2)
#         self.up_conv3 = nn.Conv2d(16,32,(3,3),1,padding=1)
#         self.up_conv4 = nn.Conv2d(32,1,(1,1),1)
#         #self.conv1_trans = nn.ConvTranspose2d(100,1,(3,3),1)
#
#
#     # 激活函数既可以使用nn，又可以调用nn.functional
#     def forward(self, x_1):
#
#         #模糊图片输入，提取模糊核信息
#         out = self.conv1(x_1)
#         out = self.pool1(out) #128
#         out = self.bn1(out)
#         out = F.relu(out)  # # 激活函数，直接调用torch.nn.functional中集成好的Relu
#         out = self.conv2(out) #124
#         out = self.pool2(out) #62
#         out = F.relu(out)
#         out = self.conv3(out)
#         out = self.pool3(out)
#         out = self.conv4(out)
#         out = self.conv5(out)
#         out = F.relu(out)
#
#         #模糊核信息和初步提取的模糊图片信息拼接
#         out = F.interpolate(out,[9,9])
#         out = change_blur(out)
#         #downsample_x1 = F.interpolate(x_1,[32,32])
#         out = torch.cat((out,downsample_x1),dim=1)
#         #恢复图像resnetblock块
#         residual_1 = out
#
#
#         out = self.res1_conv1(out)
#         out = self.res1_bn1(out)
#         out = F.relu(out)
#
#         out = self.res1_conv2(out)
#         out = self.res1_bn2(out)
#         residual_1 = self.conv1x1_1(residual_1)
#
#         out += residual_1
#
#         #第二个resnet块
#         residual_2 = out
#         out = self.res2_conv1(out)
#         out = self.res2_bn1(out)
#         out = F.relu(out)
#
#         out = self.res2_conv2(out)
#         out = self.res2_bn2(out)
#         residual_2 = self.conv1x1_2(residual_2)
#
#         out += residual_2
#
#         #上采样
#
#         out = self.upsample_1(out)
#         out = self.up_conv1(out)
#         out = F.relu(out)
#         out = self.up_conv2(out)
#        # out = F.relu(out)
#
#
#        # out = self.upsample_2(out)
#        # out = self.up_conv3(out)
#        # out = F.relu(out)
#        # out = self.up_conv4(out)
#         out = torch.sigmoid(out)
#
#
#
#
#         return out

# 定义损失函数和优化器
net = Net().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# Training
def train(epoch,net,flag):
    print('\nEpoch: %d' % epoch)
    if flag == 1:
        net = torch.nn.DataParallel(net)  # 加载模型参数关键字不匹配
        net.load_state_dict(torch.load('./checkpoint/ckpt.pth'), False)
        print('预加载成功')
    net.train()
    train_loss = 0
    for batch_idx,(input,output) in enumerate(train_loader):
        input = input.cuda()
        output = output.cuda()
        input = Variable(input)
        output = Variable(output)

        optimizer.zero_grad()
        preds = net(input,epoch,'train')
        loss = criterion(preds,output)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        print('%.3f %.3f' % (loss.item(), train_loss / (batch_idx + 1)))



def adjust_learning_rate(optimizer,epoch):
    '''
    the learning rate multiply 0.5 every 50 epoch
    '''
    if epoch%100 ==0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.5

def MAIN():
    flag = 1 #在原来的基础上加载参数训练

    for epoch in range(epochs):
        print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())) #迭代一次的时间
        train(epoch,net,flag)
        if flag == 1:
            flag = 0
        if epoch % 25 == 0:
            test(epoch,net)
        if epoch % 25 == 0:
            predict(train_path,epoch,'predict')
            predict(test_path,epoch,'predict')
        #adjust_learning_rate(optimizer,epoch)

if __name__ == "__main__":
    MAIN()
#训练网络
#for epoch in epochs:
#    img, label = data

#    if torch.cuda.is_available():
#        img = img.cuda()
#        label = label.cuda()

#    out = model(img)
#    loss = criterion(out, label)

#    optimizer.zero_grad()
#    loss.backward()
#    optimizer.step()

#    if epoch % 50 == 0:
#        print('epoch: {}, loss: {:.4}'.format(epoch, loss.data.item()))
