import torch
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
        Tensor_belong = torch.tensor(Array_belong)
        Basic = (Tensor_value==0)+2*(Tensor_value!=0)
        Array = [Basic.copy() for _ in range(20)]
        for i in range(1,11):
            Array[i-1] += (i*(Tensor_value==i))
        Tensor = torch.tensor(Array)
        aim = torch.tensor([float(_==int(result)) for _ in range(4)])
        yield Tensor, aim
    F1.close()
    F2.close()

for A,B in Read_data():
	input()
	print(A)
