import torch
import torch.nn as nn
import numpy as np
import sample_train
import net
import matplotlib.pyplot as plt
import os

if __name__=="__main__":
    epohs = 1000
    net = net.Net().cuda()
    save_path = r"params/1dnet.pth"
    opt = torch.optim.Adam(net.parameters())
    loss_fn = nn.BCELoss()
    get_data = sample_train.Get_data()
    max_data = float(torch.max(get_data))
    data = np.array(get_data) / max_data
    if os.path.exists(save_path):
        net.load_state_dict(torch.load(save_path))
    else:
        print("NO Param")
    for epoh in range(epohs):
        a=[]
        b=[]
        c=[]
        for i in range(0,len(data)-9):
            x = data[i:i+9]
            # print(x)
            y = torch.tensor(data[i+9:i+10],dtype=torch.float32).cuda()
            xs = torch.tensor(x.reshape(-1,1,9),dtype=torch.float32).cuda()
            out = net(xs)
            output=out[0]
            loss = loss_fn(output,y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            a.append(i)
            b.append(out.item())
            c.append(y.item())
            plt.plot(a,b,"red",label="data")
            plt.plot(a,c,"blue",label="label")
            plt.legend(loc="upper left")
            plt.pause(0.01)
            plt.clf()
            print("epoh:{} loss:{}".format(epoh,loss.item()))

            # torch.save(net.state_dict(),save_path)
        plt.savefig("img/{}-figure".format(epoh))