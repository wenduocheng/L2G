""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, ks=None, ds=None, **kwargs):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=ks[0], dilation=ds[0], padding=ks[0]//2 * ds[0], bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=ks[1], dilation=ds[1], padding=ks[1]//2 * ds[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, ks=None, ds=None, **kwargs):
        super().__init__()
        self.maxp_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, ks=ks, ds=ds, **kwargs)
        )

    def forward(self, x):
        return self.maxp_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, ks=None, ds=None, **kwargs):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, ks=ks, ds=ds, **kwargs)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, ks=ks, ds=ds, **kwargs)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        # print("up:", x.shape)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.convpool = nn.Sequential(
            nn.Conv2d(in_channels, in_channels*4, kernel_size=1),
            nn.BatchNorm2d(in_channels*4),
            nn.ReLU(),
            nn.Conv2d(in_channels*4, out_channels, kernel_size=1)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        return self.pool(self.convpool(x)).squeeze()
    
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, ks=None, ds=None, **kwargs):
        super(UNet, self).__init__()
        # print(ks, ds)
        if ks == None: ks = [3] * 18
        if ds == None: ds = [1] * 18
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64, ks=ks[:2], ds=ds[:2]))
        self.down1 = (Down(64, 128, ks=ks[2:4], ds=ds[2:4]))
        self.down2 = (Down(128, 256, ks=ks[4:6], ds=ds[4:6]))
        self.down3 = (Down(256, 512, ks=ks[6:8], ds=ds[6:8]))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor, ks=ks[8:10], ds=ds[8:10]))
        self.up1 = (Up(1024, 512 // factor, bilinear, ks=ks[10:12], ds=ds[10:12]))
        self.up2 = (Up(512, 256 // factor, bilinear, ks=ks[12:14], ds=ds[12:14]))
        self.up3 = (Up(256, 128 // factor, bilinear, ks=ks[14:16], ds=ds[14:16]))
        self.up4 = (Up(128, 64, bilinear, ks=ks[16:18], ds=ds[16:18]))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    # def use_checkpointing(self):
    #     self.inc = torch.utils.checkpoint(self.inc)
    #     self.down1 = torch.utils.checkpoint(self.down1)
    #     self.down2 = torch.utils.checkpoint(self.down2)
    #     self.down3 = torch.utils.checkpoint(self.down3)
    #     self.down4 = torch.utils.checkpoint(self.down4)
    #     self.up1 = torch.utils.checkpoint(self.up1)
    #     self.up2 = torch.utils.checkpoint(self.up2)
    #     self.up3 = torch.utils.checkpoint(self.up3)
    #     self.up4 = torch.utils.checkpoint(self.up4)
    #     self.outc = torch.utils.checkpoint(self.outc)

class UNet_small(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, ks=None, ds=None, **kwargs):
        super(UNet_small, self).__init__()
        if ks == None: ks = [3] * 18
        if ds == None: ds = [1] * 18
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        self.inc = (DoubleConv(n_channels, 25, ks=ks[:2], ds=ds[:2]))
        self.down1 = (Down(25, 25, ks=ks[2:4], ds=ds[2:4]))
        self.down2 = (Down(25, 25, ks=ks[4:6], ds=ds[4:6]))
        self.down3 = (Down(25, 25, ks=ks[6:8], ds=ds[6:8]))
        
        self.down4 = (Down(25, 50 // factor, ks=ks[8:10], ds=ds[8:10]))
        self.up1 = (Up(50, 50 // factor, bilinear, ks=ks[10:12], ds=ds[10:12]))
        self.up2 = (Up(50, 50 // factor, bilinear, ks=ks[12:14], ds=ds[12:14]))
        self.up3 = (Up(50, 50 // factor, bilinear, ks=ks[14:16], ds=ds[14:16]))
        self.up4 = (Up(50, 25, bilinear, ks=ks[16:18], ds=ds[16:18]))
        self.outc = (OutConv(25, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        # print("down1:", x2.shape)
        x3 = self.down2(x2)
        # print("down2:", x3.shape)
        x4 = self.down3(x3)
        # print("down3:", x4.shape)
        x5 = self.down4(x4)
        # print("down4:", x5.shape)
        x = self.up1(x5, x4)
        # print("up1:", x.shape)
        x = self.up2(x, x3)
        # print("up2:", x.shape)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits