import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    Double Convolution Block

    Conv → BatchNorm → ReLU
    Conv → BatchNorm → ReLU
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class EncoderBlock(nn.Module):
    """
    Encoder Block

    DoubleConv
        ↓
    MaxPool
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = DoubleConv(in_channels, out_channels)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):

        features = self.conv(x)

        pooled = self.pool(features)

        return features, pooled
class DecoderBlock(nn.Module):
    """
    Decoder Block

    TransposeConv
        ↓
    Concatenate Skip Features
        ↓
    DoubleConv
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        self.conv = DoubleConv(
            out_channels + skip_channels,
            out_channels
        )

    def forward(self, x, skip):

        x = self.up(x)

        x = torch.cat([x, skip], dim=1)

        x = self.conv(x)

        return x
class Encoder(nn.Module):
    """
    Shared Encoder used for both input images.
    """

    def __init__(self):
        super().__init__()

        self.enc1 = EncoderBlock(3, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)

        self.bottleneck = DoubleConv(512, 1024)

    def forward(self, x):

        f1, p1 = self.enc1(x)

        f2, p2 = self.enc2(p1)

        f3, p3 = self.enc3(p2)

        f4, p4 = self.enc4(p3)

        bottleneck = self.bottleneck(p4)

        return bottleneck, [f1, f2, f3, f4]
class FCSiamConc(nn.Module):

    def __init__(self):
        super().__init__()

        # Shared encoder
        self.encoder = Encoder()

        # Decoder
        self.dec4 = DecoderBlock(
            in_channels=2048,
            skip_channels=1024,
            out_channels=512
        )

        self.dec3 = DecoderBlock(
            in_channels=512,
            skip_channels=512,
            out_channels=256
        )

        self.dec2 = DecoderBlock(
            in_channels=256,
            skip_channels=256,
            out_channels=128
        )

        self.dec1 = DecoderBlock(
            in_channels=128,
            skip_channels=128,
            out_channels=64
        )

        self.final = nn.Conv2d(
            64,
            1,
            kernel_size=1
        )

    def forward(self, before, after):

        bottleneck1, skips1 = self.encoder(before)

        bottleneck2, skips2 = self.encoder(after)

        x = torch.cat([bottleneck1, bottleneck2], dim=1)

        skip4 = torch.cat([skips1[3], skips2[3]], dim=1)
        skip3 = torch.cat([skips1[2], skips2[2]], dim=1)
        skip2 = torch.cat([skips1[1], skips2[1]], dim=1)
        skip1 = torch.cat([skips1[0], skips2[0]], dim=1)

        x = self.dec4(x, skip4)

        x = self.dec3(x, skip3)

        x = self.dec2(x, skip2)

        x = self.dec1(x, skip1)

        x = self.final(x)

        return x

