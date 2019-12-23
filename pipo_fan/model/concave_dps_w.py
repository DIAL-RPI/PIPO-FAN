import torch
import torch.nn as nn
import torch.nn.functional as F
from .concave_dps import ResUNet as ResUNet_0

class attention(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(attention, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
            
        )
    def forward(self,x):
        x = self.conv(x)
        return x



class ResUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(ResUNet, self).__init__()
        self.resnet = ResUNet_0(n_channels, n_classes)
        # self.catconv = cat_conv(10,n_classes)
        self.att = attention(n_classes, 1)


        self.gapool1 = nn.AvgPool2d(kernel_size=224)
        self.gapool2 = nn.MaxPool2d(kernel_size=224)

    def forward(self,x):
        a,b,c,d,e = self.resnet(x)
        
        w1 = self.att(a)
        w2 = self.att(b)
        w3 = self.att(c)
        w4 = self.att(d)
        w5 = self.att(e)
        

        w1 = self.gapool1(w1) + self.gapool2(w1)
        w2 = self.gapool1(w2) + self.gapool2(w2)
        w3 = self.gapool1(w3) + self.gapool2(w3)
        w4 = self.gapool1(w4) + self.gapool2(w4)
        w5 = self.gapool1(w5) + self.gapool2(w5)

        w = torch.cat((w1, w2, w3, w4, w5),1)
        w = torch.nn.Softmax2d()(w)
        w1 = w[:,0:1,:,:]
        w2 = w[:,1:2,:,:]
        w3 = w[:,2:3,:,:]
        w4 = w[:,3:4,:,:]
        w5 = w[:,4:5,:,:]

        fi_out = w1*a + w2*b + w3*c + w4*d + w5*e

        #softmax for uniseg
        fi_out = F.softmax(fi_out, dim=1)

        return fi_out