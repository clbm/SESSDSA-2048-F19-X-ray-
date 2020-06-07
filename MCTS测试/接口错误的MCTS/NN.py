import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    '''
    基础的残差网络，有两个卷积核，最后一次ReLU前有一次skip
    '''
    
    def __init__(self, inplanes:int, planes:int, stride = 1, downsample = None):
        super(BasicBlock, self).__init__()
        #卷积核
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                        stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                        stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
    
    def forward(self, x):
        '神经网络传递'
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out


#################
BLOCKS = 10 #残差个数
#################


class Extractor(nn.Module):
    "特征提取模型函数"
    def __init__(self, inplanes:int, outplanes:int):
        super(Extractor,self).__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=3,
                        stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes)
        
        for block in range(BLOCKS):
            setattr(self, "res{}".format(block),
                BasicBlock(outplanes, outplanes))
    
    def forward(self, x):
        x.relu(self.bn1(self.conv1(x)))
        for block in range(BLOCKS-1):
            x = getattr(self, "res{}".format(block))(x)

        feature_maps = getattr(self, "res{}".format(BLOCKS - 1))(x)
        return feature_maps



class PolicyNet(nn.Module):
    "策略网络"
    def __init__(self, inplanes, outplanes):
        super(PolicyNet, self).__init__()
        self.outplanes = outplanes
        self.conv = nn.Conv2d(inplanes, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(outplanes-1, outplanes)
    
    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(-1, self.outplanes-1)
        x = self.fc(x)
        probas = self.logsoftmax(x).exp()

        return probas


class ValueNet(nn.Module):
    "价值网络"
    def __init__(self, inplanes, outplanes):
        super(ValueNet, self).__init__()
        self.outplanes = outplanes
        self.conv = nn.Conv2d(inplanes, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc1 = nn.Linear(outplanes-1, 256)
        self.fc2 = nn.Linear(256,1)
    
    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(-1, self.outplanes-1)
        x = F.relu(self.fc1(x))
        Winning = F.tanh(self.fc2(x))
        return Winning