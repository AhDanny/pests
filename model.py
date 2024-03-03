import torch
import torch.nn as nn
from resnetA import Residual


class Model(nn.Module):
    def __init__(self,numclass):
        super().__init__()
        self.netA=Residual(3,3)
        channels=3#两条支路的分类的通道数量之和
        # self.netB=
        self.FC=nn.Linear(channels,numclass)#numclass是分类的数量

    def forward(self,x):
        x0=self.netA(x)
        # x1=self.netB(x)
        out=self.FC(x0)
        return out


if __name__=="__main__":
    x=torch.randn((32,3,195,195))
    model=Model()
    out=model.forward(x)
    print(out.shape)