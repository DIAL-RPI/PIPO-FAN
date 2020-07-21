import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
            
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class one_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(one_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1)    
        )
    def forward(self,x):
        x = self.conv(x)
        return x

class res_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(res_conv, self).__init__()
        self.conv1 = double_conv(in_ch, out_ch)
        self.bridge = one_conv(in_ch, out_ch)

    def forward(self, x):
        x1 = self.conv1(x)
        if x.shape == x1.shape:
            r = x + x1
        else:
            r = self.bridge(x) + x1
        return r

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.mpconv = double_conv(in_ch, out_ch)
        self.bridge = one_conv(in_ch, out_ch)

    def forward(self, x, y):
        x = self.pool(x)
        # Concatenation
        x_1 = torch.cat((x,y),1)
        # Summation
        # x_1 = x + y
        x_2 = self.mpconv(x_1)
        if x_1.shape == x_2.shape:
            x = x_1 + x_2
        else:
            x = self.bridge(x_1) + x_2
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)
        self.bridge = one_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x) + self.bridge(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class ResUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(ResUNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.dbconv1 = res_conv(64,128)
        self.down1 = down(128, 128)
        self.dbconv2 = res_conv(64,128)
        self.dbconv3 = res_conv(128,256)
        self.down2 = down(256, 256)
        self.dbconv4 = res_conv(64,128)
        self.dbconv5 = res_conv(128,256)
        self.dbconv6 = res_conv(256,512)
        self.down3 = down(512, 512)
        self.down4 = down(1024, 512)
        self.dbup1 = res_conv(512,256)
        self.dbup2 = res_conv(256,128)
        self.dbup3 = res_conv(128,64)
        self.dbup4 = res_conv(64,64)
        self.up1 = up(1024, 256)
        self.dbup5 = res_conv(256,128)
        self.dbup6 = res_conv(128,64)
        self.dbup7 = res_conv(64,64)
        self.up2 = up(512, 128)
        self.dbup8 = res_conv(128,64)
        self.dbup9 = res_conv(64,64)
        self.up3 = up(256, 64)
        self.dbup10 = res_conv(64,64)
        self.up4 = up(128, 64)
        self.outc1 = outconv(64, n_classes)
        self.outc2 = outconv(64, n_classes)
        self.outc3 = outconv(64, n_classes)
        self.outc4 = outconv(64, n_classes)
        self.outc = outconv(64, n_classes)
        self.pool = nn.AvgPool2d(2)
        self.unpool = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.unpool = nn.Upsample(scale_factor=2, mode='nearest')
        # self.att = res_conv(64,1)
        # self.gapool = nn.AvgPool2d(kernel_size=224)

    def forward(self, x):
        x1 = self.inc(x)
        y1 = self.pool(x)
        z1 = self.inc(y1)
        x2 = self.down1(x1, z1)
        y2 = self.pool(y1)
        z2 = self.inc(y2)
        a1 = self.dbconv1(z2)
        x3 = self.down2(x2, a1)
        y3 = self.pool(y2)
        z3 = self.inc(y3)
        a2 = self.dbconv2(z3)
        a3 = self.dbconv3(a2)
        x4 = self.down3(x3, a3)
        y4 = self.pool(y3)
        z4 = self.inc(y4)
        a4 = self.dbconv4(z4)
        a5 = self.dbconv5(a4)
        a6 = self.dbconv6(a5)
        x5 = self.down4(x4, a6)
        o1 = self.dbup1(x5)
        o1 = self.dbup2(o1)
        o1 = self.dbup3(o1)
        o1 = self.dbup4(o1)
        out1 = self.outc1(o1)
        x6 = self.up1(x5, x4)
        o2 = self.dbup5(x6)
        o2 = self.dbup6(o2)
        o2 = self.dbup7(o2)
        out2 = self.outc2(o2)
        x7 = self.up2(x6, x3)
        o3 = self.dbup8(x7)
        o3 = self.dbup9(o3)
        out3 = self.outc3(o3)
        x8 = self.up3(x7, x2)
        o4 = self.dbup10(x8)
        out4 = self.outc4(o4)
        o5 = self.up4(x8, x1)
        out5 = self.outc(o5)

        o1 = self.unpool(self.unpool(self.unpool(self.unpool(o1))))
        o2 = self.unpool(self.unpool(self.unpool(o2)))
        o3 = self.unpool(self.unpool(o3))
        o4 = self.unpool(o4)
        
    
        # w1 = self.att(o1)
        # w2 = self.att(o2)
        # w3 = self.att(o3)
        # w4 = self.att(o4)
        # w5 = self.att(o5)

        # w1 = self.gapool(w1)
        # w2 = self.gapool(w2)
        # w3 = self.gapool(w3)
        # w4 = self.gapool(w4)
        # w5 = self.gapool(w5)

        # w = torch.cat((w3, w4, w5),1)
        # w = torch.nn.Softmax2d()(w)
        # w3 = w[:,0:1,:,:]
        # w4 = w[:,1:2,:,:]
        # w5 = w[:,2:3,:,:]
        # w4 = w[:,3:4,:,:]
        # w5 = w[:,4:5,:,:]
        
        out1 = self.unpool(self.unpool(self.unpool(self.unpool(out1))))
        out2 = self.unpool(self.unpool(self.unpool(out2)))
        out3 = self.unpool(self.unpool(out3))
        out4 = self.unpool(out4)

        # out = w3*out3 + w4*out4 + w5*out5

        return out1, out2, out3, out4, out5

# class ResUNet(nn.Module):
#     def __init__(self, n_channels, n_classes):
#         super(ResUNet, self).__init__()
#         self.resnet = ResUNet_0(n_channels, n_classes)
#         # self.catconv = cat_conv(10,n_classes)
#         self.att = nn.Sequential(
#             nn.BatchNorm2d(2),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(2, 1, 1),
#             nn.BatchNorm2d(1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(1, 1, 3, padding=1)
            
#         )
#         self.gapool1 = nn.AvgPool2d(kernel_size=224)
#         self.gapool2 = nn.MaxPool2d(kernel_size=224)

#     def forward(self,x):
#         a,b,c,d,e = self.resnet(x)
        
#         w1 = self.att(a)
#         w2 = self.att(b)
#         w3 = self.att(c)
#         w4 = self.att(d)
#         w5 = self.att(e)

#         w1 = self.gapool1(w1) + self.gapool2(w1)
#         w2 = self.gapool1(w2) + self.gapool2(w2)
#         w3 = self.gapool1(w3) + self.gapool2(w3)
#         w4 = self.gapool1(w4) + self.gapool2(w4)
#         w5 = self.gapool1(w5) + self.gapool2(w5)

#         w = torch.cat((w1, w2, w3, w4, w5),1)
#         w = torch.nn.Softmax2d()(w)
#         w1 = w[:,0:1,:,:]
#         w2 = w[:,1:2,:,:]
#         w3 = w[:,2:3,:,:]
#         w4 = w[:,3:4,:,:]
#         w5 = w[:,4:5,:,:]

#         fi_out = w1*a + w2*b + w3*c + w4*d + w5*e

#         return fi_out
