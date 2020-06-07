# pylint: disable=no-member
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
class Net_d(nn.Module):
    '二通道神经网络，用于合并'
    def __init__(self):
        super(Net_d, self).__init__()
        self.conv1 = nn.Conv2d(2, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc2 = nn.Linear(32,18)
        self.fc3 = nn.Linear(18,4)
        self.PAD = lambda x:F.pad(x,(1,1,1,1),"constant",value=0)

    def forward(self, x):
        x = self.PAD(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.PAD(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

def Read_data():
    "返回训练数据与结果"
    global device
    F1 = open('./Inputs_d.txt','r')
    F2 = open('result_d.txt','r')
    get_belong = lambda x: {'+':1,'-':0}[x[0]]
    get_value = lambda x: float(x[1:])
    Continue = True
    while Continue:
        Array = []
        results = []
        for  _ in range(15):
            result = F2.readline()
            if result == '':
                F1.close()
                F2.close()
                Continue = False
                break
            results.append(int(result))
            Array_value = []
            Array_belong = []
            for _ in range(4):
                S = F1.readline().split()
                Array_value.append(list(map(get_value,S)))
                Array_belong.append(list(map(get_belong,S)))
            Array.append([Array_value,Array_belong])
        aim = torch.tensor(results)
        ARRAY = torch.tensor(Array)
        yield ARRAY, aim#.to(device)




if __name__ == "__main__":
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net_d()
    #net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    for epoch in range(10):  #训练十次

        running_loss = 0.0
        for i, data in enumerate(Read_data(), 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 3000 == 2999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 3000))
                running_loss = 0.0
        torch.save(net.state_dict(),f'./model_{epoch}')
