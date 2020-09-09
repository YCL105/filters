import torch
import numpy as np
import os
import matplotlib.pyplot as plt

from train import *

from evaluate import *

def test(epoch,net):
    print('\nTest')
    net.eval()
    test_loss = 0
    for batch_idx,(input,output) in enumerate(test_loader):
        input = input.cuda()
        output = output.cuda()
        input = Variable(input)
        output = Variable(output)
    preds = net(input,epoch,'test')
    loss = nn.MSELoss()(preds, output)
    test_loss += loss.item()
    print('%.3f %.3f' % (loss.item(), test_loss / (batch_idx + 1)))

    # Save checkpoint.
    #global best_loss
    #best_loss = 3.0
    #test_loss /= len(test_loader)
    #if test_loss < best_loss:
     #   print('Saving..')
     #   state = {
     #       'net': net.state_dict(),
     #       'loss': test_loss,
     #       'epoch': epoch,
     #   }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(net, './checkpoint/net.pth')
        #best_loss = test_loss

#调用模型预测并评价
def predict(filename,epoch,name):
    #加载模型

   # model = torch.nn.DataParallel(model)  #加载模型参数关键字不匹配
    model = torch.load('./checkpoint/net.pth')

    model.train()
    #预测并输出图片
    size = 256
    x_data,y_data,List = ImageReader(filename)  #读取出train图片
    x_data = x_data.cuda()
    m = x_data.shape
    for i in range(m[0]): #遍历每个图片
        start = time.time()
        pred = model(x_data[i,:,:,:].reshape(1,1,size,size),epoch,name)
        n_pred = pred.cpu().detach().numpy()
        n_pred = n_pred.reshape(m[1],m[2],m[3])
        n_pred = np.transpose(n_pred,(1,2,0))

        cv2.imwrite(filename+'/final_results/'+List[i], n_pred*255.)
        #cv2.imwrite(filename  + List[i], n_pred * 255.)
        end = time.time()
    #给出评价
        img1 = x_data[i,:,:,:].cpu().detach().numpy()#tensor B,C,H,W->H,W np
        img1 = img1.reshape(size,size)

        img2 = y_data[i,:,:,:].cpu().detach().numpy()  #tensor B,C,H,W->H,W np
        img2 = img2.reshape(size, size)

        img3 = n_pred.reshape(size,size)

        evlauate_metrics(img1*255., img2*255., img3*255., (end - start),List[i])
