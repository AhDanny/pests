import torch
import torch.nn as nn
from resnetA import Residual

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.netA=Residual(3,3)
        # self.netB=
    def forward(self,x):
        x0=self.netA(x)
        # x1=self.netB(x)
        return x0


if __name__=="__main__":
    x=torch.randn((32,3,195,195))
    model=Model()
    out=model.forward(x)
    print(out.shape)