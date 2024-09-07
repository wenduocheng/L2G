import torch
import torch.nn as nn


# ref: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py

class DoubleConv1D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, ks, ds):
        super(DoubleConv1D, self).__init__()
        k1, d1 = ks[0], ds[0]
        k2, d2 = ks[1], ds[1]
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=k1, dilation=d1, padding=(k1//2) * d1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=k2, dilation=d2, padding=(k2//2) * d2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down1D(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, ks, ds):
        super(Down1D, self).__init__()
        self.max_conv = nn.Sequential(   # change the name from maxpool_conv to conv to avoid being removed in dash.py
            nn.MaxPool1d(2),
            DoubleConv1D(in_channels, out_channels, ks, ds)
        )

    def forward(self, x):
        return self.max_conv(x)  


class Up1D(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, ks, ds, bilinear=True):
        super(Up1D, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
            self.conv = DoubleConv1D(in_channels, out_channels, ks, ds)
        else:
            self.up = nn.ConvTranspose1d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv1D(in_channels, out_channels, ks, ds)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CWH
        diffY = x2.size()[2] - x1.size()[2]

        x1 = nn.functional.pad(x1, [diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# class OutConv1D(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(OutConv1D, self).__init__()
#         self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

#     def forward(self, x):
#         return self.conv(x)


class UNet1D(nn.Module):
    def __init__(self, n_channels, num_classes, ks=None, ds=None, bilinear=True):
        super(UNet1D, self).__init__()
        self.n_channels = n_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        if ks is None:
            ks = [3] * 18
        if ds is None:
            ds = [1] * 18
        print('ks: ', ks)
        print('ds: ', ds)
        self.inc = DoubleConv1D(n_channels, 64, ks[0:2], ds[0:2])
        self.down1 = Down1D(64, 128, ks[2:4], ds[2:4])
        self.down2 = Down1D(128, 256, ks[4:6], ds[4:6])
        self.down3 = Down1D(256, 512, ks[6:8], ds[6:8])
        factor = 2 if bilinear else 1
        self.down4 = Down1D(512, 1024 // factor, ks[8:10], ds[8:10])
        self.up1 = Up1D(1024, 512 // factor, ks[10:12], ds[10:12], bilinear)
        self.up2 = Up1D(512, 256 // factor, ks[12:14], ds[12:14], bilinear)
        self.up3 = Up1D(256, 128 // factor, ks[14:16], ds[14:16], bilinear)
        self.up4 = Up1D(128, 64, ks[16:18], ds[16:18], bilinear)
  
        
        # final prediction
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling to ensure fixed output size
        self.fc = nn.Linear(64, self.num_classes)  # Fully connected layer for classification

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

        x = self.global_pool(x)  # Apply global average pooling
        x = x.squeeze(-1)  # Remove the last dimension 
        logits = self.fc(x)   
        return logits


