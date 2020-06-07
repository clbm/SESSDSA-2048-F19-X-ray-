# pylint: disable=no-member
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from random import shuffle
class Net_d(nn.Module):
    '二通道神经网络，用于合并'
    def __init__(self):
        super(Net_d, self).__init__()
        self.conv1 = nn.Conv2d(10, 80, (4,1))
        self.pool1 = nn.MaxPool2d((1, 2))
        self.conv2 = nn.Conv2d(10,160,(1,8))
        self.pool2 = nn.MaxPool2d((2,1))
        self.fc = nn.Linear(640,4)

    def forward(self, x):
        x1 = self.pool1(F.relu(self.conv1(x)))
        x2 = self.pool2(F.relu(self.conv2(x)))
        x1 = x1.view(-1,320)
        x2 = x2.view(-1,320)
        x = torch.cat((x1,x2),1)
        return self.fc(x)


def Read_data():
    "返回训练数据与结果"
    global device
    F1 = open('./Inputs_d.txt','r')
    F2 = open('result_d.txt','r')
    get_belong = lambda x: {'+':1,'-':-1}[x[0]]
    get_value = lambda x: float(x[1:])
    for result in F2.readlines():
        Array_value = []
        Array_belong = []
        for _ in range(4):
            S = F1.readline().split()
            Array_value.append(list(map(get_value,S)))
            Array_belong.append(list(map(get_belong,S)))
        Tensor_value = torch.tensor(Array_value).numpy()
        Tensor_belong = torch.tensor(Array_belong).numpy()
        Array = [(Tensor_value==0)+2*(Tensor_value!=0) for _ in range(10)]
        for i in range(1,11):
            Array[i-1] += i*(Tensor_value==i)
        Tensor = torch.tensor(Array).numpy()*Tensor_belong
        yield Tensor, int(result)
    F1.close()
    F2.close()

def make_barch(Iter):
    "返回训练数据与结果"
    Continue = True
    while Continue:
        Array = []
        results = []
        for  _ in range(15):
            if Iter.__length_hint__()==0:
                break
                Continue = False
            array,correct = next(Iter)
            Array.append(array)
            results.append(correct)
        aim = torch.tensor(results)
        ARRAY = torch.FloatTensor(Array)
        if Continue:
            yield ARRAY, aim#.to(device)




if __name__ == "__main__":
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('import done')
    net = Net_d()
    print('net done')
    #net.to(device)
    Data_pair = list(Read_data())
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    print('start training')
    for epoch in range(10):  #训练十次

        running_loss = 0.0
        shuffle(Data_pair)
        for i, data in enumerate(make_barch(iter(Data_pair)), 0):
            # get the inputs
            inputs, labels = data
            if len(inputs.shape) == 1:
                break
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            #print(inputs)
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
        torch.save(net.state_dict(),f'./model_{epoch+1}')
