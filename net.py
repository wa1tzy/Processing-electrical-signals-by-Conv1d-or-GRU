import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1,32,3,2,1),
            nn.BatchNorm1d(32),
            nn.ReLU()#5
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32,64,3,2,1),
            nn.BatchNorm1d(64),
            nn.ReLU()#3
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64,128,3,1,0),
            # nn.BatchNorm1d(128),
            nn.ReLU()#1
        )
        self.fc = nn.Sequential(
            nn.Linear(128,1),
            nn.Sigmoid()#1
        )
    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2)
        y3 = y3.reshape(-1,128)
        y4 = self.fc(y3)
        return y4
if __name__ == '__main__':
    import torch
    # data = torch.randint(10,100,(1000,),dtype=torch.float32)
    # data2 = torch.randint(-3,3,(1000,),dtype=torch.float32)
    # test_data = data+data2
    # print(data2)
    # x = torch.randint(10,100,(1,1,9),dtype=torch.float32)
    # print(x)
    # net = Net()
    # y = net.forward(x)
    # print(y)
    x = torch.tensor([[[1.,2.,3.,4.,5.,6.,7.,8.,9.,]]])
    net=Net()
    print(net(x))