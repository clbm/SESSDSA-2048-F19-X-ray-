# pylint: disable=no-member
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time

#以下为官网上的原文和谷歌浏览器的翻译。
#机翻可能有不准，仅供参考。

#torch.Tensor – A multi-dimensional array with support for autograd operations like backward(). Also holds the gradient w.r.t. the tensor.
#torch.Tensor –一个多维数组，支持诸如backward()之类的autograd操作。还保持张量的梯度。

#nn.Module – Neural network module. Convenient way of encapsulating parameters, with helpers for moving them to GPU, exporting, loading, etc.
#nn.Module –神经网络模块。方便的封装参数的方式，并带有将它们移动到GPU，导出，加载等的辅助工具。

#nn.Parameter – A kind of Tensor, that is automatically registered as a parameter when assigned as an attribute to a Module.
#nn.Parameter –一种Tensor，在将其作为属性分配给Module时会自动注册为参数。

#autograd.Function – Implements forward and backward definitions of an autograd operation. 
# Every Tensor operation, creates at least a single Function node, that connects to functions that created a Tensor and encodes its history.
#autograd.Function –实现对autograd操作的正向和反向定义。每个Tensor操作都会创建至少一个Function节点，该节点连接到创建Tensor并对其历史进行编码的函数。


class Net(nn.Module):

    def __init__(self):
        super(Net,self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        #第一个卷积核单通道输入，6通道输出，5*5的矩阵
        #第二个卷积核6通道输入，16通道输出，5*5的矩阵
        # kernel
        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)) ,2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self,x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_feature = 1
        for s in size:
            num_feature *= s
        return num_feature


net = Net()
with torch.no_grad():
    t1 = time()
    for _ in range(1000):
        Input = torch.randn(1,1,32,32)
        out = net(Input)
    t2 = time()
    print(t2-t1)