# import device as device
import torch.nn as nn
from src.models import network
class OfficeHome_Shot(nn.Module):
    def __init__(self,netF,netC):
        super(OfficeHome_Shot, self).__init__()
        self.netF = netF
        self.netC = netC
        pass

    def forward(self, x):
        x = self.netF(x)
        # x = self.netB(x)
        x = self.netC(x)
        return x

class OfficeHome_Shot_2(nn.Module):
    def __init__(self,netF,netB,netC):
        super(OfficeHome_Shot_2, self).__init__()
        self.netF = netF
        self.netB = netB
        self.netC = netC
        pass

    def forward(self, x):
        x = self.netF(x)
        x = self.netB(x)
        x = self.netC(x)
        return x
